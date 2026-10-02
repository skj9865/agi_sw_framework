import torch
import torch.nn as nn
from sj_compat import DropPath
from sj_compat import LIFNode, ParametricLIFNode

import torch.nn.functional as F
import math
from inspect import isfunction


MIN_EXPERT_CAPACITY = 4


def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)


def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)


class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_


class MembraneMixing(nn.Module):
    """Pulls each token's membrane potential toward the spatial mean.

        h_out = (1 - α) * h + α * mean(h over H,W)

    Higher α → more homogeneous potentials → more similar spikes →
    router selects fewer unique experts.

    `mix_ratio` is the *initial* effective α (mapped through sigmoid internally
    so the learnable parameter is unconstrained).
    """
    def __init__(self, mix_ratio=0.1, learnable=True):
        super().__init__()
        init_logit = math.log(mix_ratio / (1.0 - mix_ratio))
        if learnable:
            self._mix_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.register_buffer('_mix_logit', torch.tensor(init_logit))

    @property
    def alpha(self):
        return torch.sigmoid(self._mix_logit)

    def forward(self, h):
        alpha = self.alpha
        h_mean = h.mean(dim=(-2, -1), keepdim=True)
        return (1 - alpha) * h + alpha * h_mean


class MultiStepMixingLIFNode(LIFNode):
    """LIFNode (step_mode='m') with membrane-potential mixing between charge and fire.

    Subclasses spikingjelly's LIFNode so neuronal_charge,
    neuronal_fire, neuronal_reset, surrogate function, state management,
    and reset() are all inherited exactly.

    Only overrides forward() to insert mixing into the timestep loop.
    Uses backend='torch' (mixing breaks cupy kernel fusion).
    """
    def __init__(self, tau: float = 2.0, decay_input: bool = True,
                 v_threshold: float = 1.0, v_reset: float = 0.0,
                 detach_reset: bool = True,
                 mix_ratio: float = 0.1, learnable_mix: bool = True):
        super().__init__(tau=tau, decay_input=decay_input,
                         v_threshold=v_threshold, v_reset=v_reset,
                         detach_reset=detach_reset, backend='torch', step_mode='m')
        self.mixing = MembraneMixing(mix_ratio=mix_ratio, learnable=learnable_mix)

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        self.v_float_to_tensor(x_seq[0])
        spike_seq = []
        self.v_seq = []
        for t in range(x_seq.shape[0]):
            self.neuronal_charge(x_seq[t])
            self.v = self.mixing(self.v)      # ← membrane mixing
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            spike_seq.append(spike.unsqueeze(0))
            self.v_seq.append(self.v.unsqueeze(0))
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.cat(self.v_seq, 0)
        return spike_seq


class Erode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


class MS_MLP_Expert(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        spike_mode="lif",
        layer=0,
        tau=2.0,
        use_output_lif=False,


    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)

        self.fc1_bn = nn.BatchNorm2d(hidden_features)


        if spike_mode == "lif":
            self.fc1_lif = LIFNode(tau=tau, detach_reset=True, backend="cupy", step_mode='m')
        elif spike_mode == "plif":
            self.fc1_lif = ParametricLIFNode(
                init_tau=tau, detach_reset=True, backend="cupy", step_mode='m'
            )

        self.fc2_conv = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1
        )

        self.fc2_bn = nn.BatchNorm2d(out_features)

        if spike_mode == "lif":
            self.fc2_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')
        elif spike_mode == "plif":
            self.fc2_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy", step_mode='m'
            )

        self.use_output_lif = use_output_lif
        if use_output_lif:
            if spike_mode == "lif":
                self.output_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')
            elif spike_mode == "plif":
                self.output_lif = ParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x

        x = self.fc1_lif(x)
        self.last_fc1_spikes = x.detach()
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()


        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()


        if self.use_output_lif:
            x = self.output_lif(x)

        return x, hook


class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        if dvs:
            self.pool = Erode()
        self.scale = 0.125
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.q_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')
        elif spike_mode == "plif":
            self.q_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy", step_mode='m'
            )

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.k_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')
        elif spike_mode == "plif":
            self.k_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy", step_mode='m'
            )

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.v_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')
        elif spike_mode == "plif":
            self.v_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy", step_mode='m'
            )

        if spike_mode == "lif":
            self.attn_lif = LIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy", step_mode='m'
            )
        elif spike_mode == "plif":
            self.attn_lif = ParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy", step_mode='m'
            )

        self.talking_heads = nn.Conv1d(
            num_heads, num_heads, kernel_size=1, stride=1, bias=False
        )
        if spike_mode == "lif":
            self.talking_heads_lif = LIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy", step_mode='m'
            )
        elif spike_mode == "plif":
            self.talking_heads_lif = ParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy", step_mode='m'
            )

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)

        if spike_mode == "lif":
            self.shortcut_lif = LIFNode(
                tau=2.0, detach_reset=True, backend="cupy", step_mode='m'
            )
        elif spike_mode == "plif":
            self.shortcut_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy", step_mode='m'
            )

        self.mode = mode
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x
        N = H * W
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)

        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        q = (
            q_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()
        k = (
            k_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()
        v = (
            v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B head N C//h

        kv = k.mul(v)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv_before"] = kv
        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T, B, C, H, W)
            .contiguous()
        )

        x = x + identity
        return x, v, hook


class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        top_k = 1,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 4.,
        capacity_factor_eval = 4.,
        mixing_mode = 'post_linear',
        mix_ratio = 0.1,
        sample_routing = False,
        use_ste = False,
        early_exit = False,
        exit_threshold = 0.5,
        exit_low_T = 1,
        ):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.top_k = top_k
        self.mixing_mode = mixing_mode
        self.sample_routing = sample_routing
        self.use_ste = use_ste
        self.early_exit = early_exit
        self.exit_threshold = exit_threshold
        self.exit_low_T = exit_low_T  # accepted for signature compat; unused here
        self.last_gate_logits = None
        self.last_gate_input = None    # raw feature map fed INTO the router (entropy source)
        self.last_gate_spikes = None   # gate_lif1 spike output, for activity analysis

        router_hidden = dim
        if mixing_mode == 'membrane':
            self.gate_lif1 = MultiStepMixingLIFNode(tau=2.0, detach_reset=True, mix_ratio=mix_ratio)
        else:
            self.gate_lif1 = LIFNode(tau=2.0, detach_reset=True, backend='torch', step_mode='m')
        self.gate_fc1 = nn.Linear(dim, num_gates, bias=False)
        if mixing_mode in ('post_linear', 'grouped'):
            self.mix_ratio = mix_ratio  # fixed, not learnable — safe to load from any checkpoint
        if mixing_mode == 'grouped':
            self.gate_thresh = nn.Linear(dim, 1)
            self.group_temp = 1.0  # sigmoid sharpness (scores are standardized → ~unit scale)
            nn.init.zeros_(self.gate_thresh.weight)
            nn.init.zeros_(self.gate_thresh.bias)


        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

        self.last_indices = None
        self.last_masks = None
        self.last_gates = None
        self.last_raw_gates = None


    def _grouped_mix(self, x_gate, spike_feats):
        feats = spike_feats.flatten(3)                       # (T, B, C, N)
        counts = feats.sum(dim=(0, 2))                       # (B, N) firing per token
        img_feat = feats.mean(dim=(0, 3))                    # (B, C) per-image summary

        mean = counts.mean(dim=1, keepdim=True)
        std = counts.std(dim=1, keepdim=True).clamp(min=1e-6)
        scores = (counts - mean) / std                       # (B, N), ~unit scale

        tau = self.gate_thresh(img_feat).squeeze(-1)         # (B,) learnable per-image threshold
        g = torch.sigmoid((scores - tau[:, None]) / self.group_temp)  # (B, N) soft "high" weight

        def group_mean(w):                                   # w: (B, N) soft membership
            denom = w.sum(dim=1).clamp(min=1e-6)             # (B,)
            s = torch.einsum('tbne,bn->tbe', x_gate, w)      # (T, B, E)
            return s / denom[None, :, None]                  # (T, B, E)

        high_mean = group_mean(g)                            # (T, B, E)
        low_mean = group_mean(1.0 - g)                       # (T, B, E)
        token_mean = (g[None, :, :, None] * high_mean[:, :, None, :]
                      + (1 - g)[None, :, :, None] * low_mean[:, :, None, :])  # (T, B, N, E)

        r = self.mix_ratio
        return (1 - r) * x_gate + r * token_mean

    def forward(self, x, importance=None):
        T, B, dim, H, W = x.shape
        group_size = H * W
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval


        self.last_gate_input = x     # (T, B, C, H, W) raw feature map fed INTO the router
        x = self.gate_lif1(x)        # (T, B, C, H, W)
        x_gate = x.flatten(3).permute(0, 1, 3, 2).contiguous()  # T, B, N, C
        x_gate = self.gate_fc1(x_gate)         # (T, B, N, E)
        if self.mixing_mode == 'post_linear':
            x_gate = (1 - self.mix_ratio) * x_gate + self.mix_ratio * x_gate.mean(dim=2, keepdim=True)
        elif self.mixing_mode == 'grouped':
            x_gate = self._grouped_mix(x_gate, spike_feats=x)
        self.last_gate_logits = x_gate
        self.last_gate_spikes = x        # (T, B, C, H, W) gate_lif1 spikes (token activity)
        x_pooled = x_gate.mean(dim=0) if self.early_exit else x_gate.mean(dim=0)  # B, N, E

        raw_gates = x_pooled.softmax(dim=-1)


        masks = []
        gates = []
        indices = []
        positions = []
        
        gates_remaining = raw_gates.clone()
        cumulative_mask_count = torch.zeros(B, 1, num_gates).to(raw_gates.device)  # Track capacity usage

        use_sampling = self.sample_routing and self.training
        if use_sampling:
            B_, N_, E_ = raw_gates.shape
            probs_flat = raw_gates.reshape(-1, E_)
            sampled_flat = torch.multinomial(probs_flat, num_samples=self.top_k, replacement=False)
            sampled_indices = sampled_flat.reshape(B_, N_, self.top_k)  # B, N, top_k

        for k in range(self.top_k):
            if use_sampling:
                index_k = sampled_indices[..., k]
                gate_k = raw_gates.gather(-1, index_k.unsqueeze(-1)).squeeze(-1)
            else:
                gate_k, index_k = top1(gates_remaining)  # B, N
            mask_k_hard = F.one_hot(index_k, num_gates).float()  # B, N, E
            if self.use_ste and self.training:
                soft_dist = gates_remaining if k > 0 else raw_gates
                mask_k = soft_dist + (mask_k_hard - soft_dist).detach()
            else:
                mask_k = mask_k_hard
            
            if importance is not None:
                if k == 0:
                    importance_mask = (importance == 1.).float()
                else:
                    importance_mask = (importance > 0.).float()
                
                mask_k *= importance_mask[..., None]
                gate_k *= importance_mask
            
            position_k = cumsum_exclusive(mask_k, dim=-2) + cumulative_mask_count
            position_k = position_k * mask_k
            
            cumulative_mask_count = cumulative_mask_count + mask_k.sum(dim=-2, keepdim=True)
            
            masks.append(mask_k)
            gates.append(gate_k)
            indices.append(index_k)
            positions.append(position_k.sum(dim=-1))  # B, N
            
            gates_remaining = gates_remaining * (1. - mask_k)
        
        if self.top_k > 1:
            gate_sum = sum(gates) + self.eps
            gates = [g / gate_sum for g in gates]
        else:
            gates = [g + (1.0 - g).detach() for g in gates]

        density_1 = masks[0].mean(dim=-2)
        density_1_proxy = raw_gates.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        if self.top_k > 1:
            for k in range(1, self.top_k):
                if policy == "all":
                    pass
                elif policy == "none":
                    masks[k] = torch.zeros_like(masks[k])
                elif policy == "threshold":
                    masks[k] *= (gates[k] > threshold).float().unsqueeze(-1)
                elif policy == "random":
                    probs = torch.zeros_like(gates[k]).uniform_(0., 1.)
                    masks[k] *= (probs < (gates[k] / max(threshold, self.eps))).float().unsqueeze(-1)
        
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)
        
        cumulative_position = torch.zeros(B, 1, num_gates).to(raw_gates.device)
        for k in range(self.top_k):
            position_k = cumsum_exclusive(masks[k], dim=-2) + cumulative_position
            position_k = position_k * masks[k]
            
            masks[k] *= (position_k < expert_capacity_f).float()
            
            mask_k_flat = masks[k].sum(dim=-1)
            positions[k] = position_k.sum(dim=-1)
            gates[k] *= mask_k_flat
            
            cumulative_position = cumulative_position + masks[k].sum(dim=-2, keepdim=True)
        
        combine_tensor = torch.zeros(B, group_size, num_gates, expert_capacity).to(raw_gates.device)
        
        for k in range(self.top_k):
            mask_k_flat = masks[k].sum(dim=-1)
            combine_tensor += (
                gates[k][..., None, None]
                * mask_k_flat[..., None, None]
                * F.one_hot(indices[k], num_gates)[..., None]
                * safe_one_hot(positions[k].long(), expert_capacity)[..., None, :]
            )
        
        dispatch_tensor = combine_tensor.bool().to(combine_tensor)

        self.last_indices = indices
        self.last_masks = masks
        self.last_gates = gates
        self.last_raw_gates = raw_gates

        return dispatch_tensor, combine_tensor, loss


class ExpertEarlyExit(nn.Module):
    """Per-expert-group, entropy-gated timestep budget (inference only).

    Tokens are grouped by the expert the router assigned them to. For each
    group we compute the mean entropy of the router's softmax over experts.
    Entropy is standardized per layer (z-score over that layer's tokens) so the
    same threshold is comparable across layers, whose absolute entropy levels
    differ. The group then runs `high_T` timesteps when its (normalized) entropy
    is above `entropy_threshold` and `low_T` below it — i.e. uncertain routing
    -> more timesteps, confident routing -> fewer. Set `high_entropy_high_T =
    False` to invert. If `prune_threshold` is set, a group whose normalized
    entropy falls below it is pruned entirely (skipped, zero contribution).
    """
    def __init__(self, num_experts, entropy_threshold=0.5, high_T=None, low_T=1,
                 high_entropy_high_T=True, prune_threshold=None, per_layer_norm=True,
                 metric='entropy', exit_mode='absolute'):
        super().__init__()
        self.num_experts = num_experts
        self.entropy_threshold = entropy_threshold
        self.high_T = high_T   # None -> full T resolved at call time
        self.low_T = low_T
        self.high_entropy_high_T = high_entropy_high_T
        self.prune_threshold = prune_threshold   # None -> no pruning
        self.per_layer_norm = per_layer_norm     # z-score per layer (absolute mode)
        assert metric in ('entropy', 'activity', 'both'), f"metric must be entropy|activity|both, got {metric}"
        self.metric = metric
        assert exit_mode in ('absolute', 'relmax'), f"exit_mode must be absolute|relmax, got {exit_mode}"
        self.exit_mode = exit_mode
        self.last_t_e = None
        self.last_entropy = None       # normalized per-group value used for decisions
        self.last_entropy_raw = None   # raw per-group value for reference
        self.last_pruned = None        # list of pruned expert ids

    @torch.no_grad()
    def forward(self, gate_input, gate_spikes, index):
        T, B, C, H, W = gate_input.shape
        E = self.num_experts
        high_T = T if self.high_T is None else min(self.high_T, T)
        low_T = min(self.low_T, T)

        feat = gate_input.mean(dim=0).flatten(2)           # (B, C, N)
        p = feat.softmax(dim=1)
        ent = -(p * p.clamp_min(1e-9).log()).sum(dim=1)    # (B, N) entropy (nats)
        ent_raw_per_tok = ent / math.log(C)                # (B, N) entropy in [0, 1]
        sa_raw_per_tok = gate_spikes.float().mean(dim=(0, 2)).flatten(1)   # (B, N)

        def zscore_img(x):  # per-IMAGE z-score over that image's tokens (dim=1)
            m = x.mean(dim=1, keepdim=True)
            s = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            return (x - m) / s

        if self.metric == 'entropy':
            raw = ent_raw_per_tok
            val = zscore_img(ent) if self.per_layer_norm else ent_raw_per_tok
        elif self.metric == 'activity':
            raw = sa_raw_per_tok
            val = zscore_img(sa_raw_per_tok) if self.per_layer_norm else sa_raw_per_tok
        else:  # 'both' -> average per-image z-scores
            val = 0.5 * (zscore_img(ent) + zscore_img(sa_raw_per_tok))
            raw = 0.5 * (ent_raw_per_tok + sa_raw_per_tok)

        member = F.one_hot(index, E).to(ent.dtype)         # (B, N, E)
        counts_b = member.sum(dim=1)                       # (B, E) tokens/expert/image
        denom_b = counts_b.clamp(min=1.0)
        active = counts_b > 0                              # (B, E)
        val_e = torch.einsum('bn,bne->be', val, member) / denom_b              # (B, E)
        raw_e = torch.einsum('bn,bne->be', raw, member) / denom_b              # (B, E)
        sa_e  = torch.einsum('bn,bne->be', sa_raw_per_tok, member) / denom_b   # (B, E)

        if self.exit_mode == 'relmax':
            masked = torch.where(active, sa_e, sa_e.new_full((), -1.0))
            max_b = masked.max(dim=1, keepdim=True).values.clamp_min(1e-9)     # (B, 1)
            decision = sa_e / max_b                        # (B, E) in (0, 1]
            self.last_entropy = decision.detach()          # ratio used for the decision
            self.last_entropy_raw = sa_e.detach()
        else:
            decision = val_e                               # (B, E)
            self.last_entropy = val_e.detach()
            self.last_entropy_raw = raw_e.detach()

        is_high = decision >= self.entropy_threshold       # (B, E)
        if self.exit_mode != 'relmax' and not self.high_entropy_high_T:
            is_high = ~is_high
        high_t = torch.full_like(counts_b, float(high_T))
        low_t = torch.full_like(counts_b, float(low_T))
        t_e = torch.where(is_high, high_t, low_t)          # (B, E)

        pruned_mask = torch.zeros_like(active)             # (B, E) bool
        if self.prune_threshold is not None:
            pruned_mask = active & (decision < self.prune_threshold)
            t_e = torch.where(pruned_mask, low_t, t_e)
        t_e = torch.where(active, t_e, torch.ones_like(t_e)).long()   # empty group -> 1

        self.last_t_e = t_e                 # (B, E) per-image timestep budget
        self.last_pruned = pruned_mask      # (B, E) bool
        return t_e


class ExpertTimestepPredictor(nn.Module):
    """Input-aware per-expert timestep budget predictor (DT-SNN / SEENN, per-expert).

    DT-SNN (DAC'23) and SEENN (NeurIPS'23) both treat the number of timesteps as a
    variable *conditioned on the input*, deciding it per-sample for the whole network
    from classifier confidence. There is no per-expert classifier confidence, so here a
    tiny learned head reads each expert's dispatched INPUT (pooled over timesteps and
    tokens, per image) and predicts a categorical distribution q_e over candidate
    budgets t_e in {min_T, ..., T}. This generalizes the input-aware idea to per-expert
    granularity (SEENN-II predicts T from input via RL; we do it feed-forward and
    differentiable, trained end-to-end under a compute/ponder penalty).

    Training (differentiable, full-T compute): the expert is run for the full T steps;
    the effective output is the EXPECTATION over freeze-tail variants weighted by q_e,
        eff = sum_te q_e(te) * freeze(out, te),
    where freeze(out, te) repeats the te-th timestep for all later steps (same repeat-last
    rule as `_apply_per_image_te`). The ponder cost rho_e = sum_te q_e(te) * te (expected
    number of steps) is returned so the loss can add lambda * rho_e.

    Inference (hard, real savings): te* = argmax_t q_e(t) per image; tail frozen past te*
    via `_apply_per_image_te` (at batch=1 deployment the kernel can be sliced to te* steps).
    The head is a non-spiking control signal (standard for ACT / SEENN policies).
    """
    def __init__(self, dim, T, min_T=1, hidden=None):
        super().__init__()
        self.T = T
        self.min_T = max(1, min(min_T, T))
        self.budgets = list(range(self.min_T, T + 1))   # candidate t_e values
        hidden = hidden or max(8, dim // 4)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(self.budgets)),
        )
        self.register_buffer('budget_vals', torch.tensor(self.budgets, dtype=torch.float32))
        self.last_t_e = None   # (B,) per-image budget from the most recent call (float)

    def _logits(self, expert_input):
        s = expert_input.mean(dim=(0, 3, 4))   # (B, D)
        return self.net(s)                     # (B, K)

    def forward(self, expert_input, out_full):
        logits = self._logits(expert_input)              # (B, K)
        if self.training:
            q = logits.softmax(dim=-1)                   # (B, K)
            eff = torch.zeros_like(out_full)
            for k, te in enumerate(self.budgets):
                eff = eff + q[:, k].view(1, -1, 1, 1, 1) * _freeze_at(out_full, te)
            ponder = (q * self.budget_vals.view(1, -1)).sum(dim=-1)   # (B,) expected #steps
            self.last_t_e = ponder.detach()
            return eff, ponder
        else:
            te_b = self.budget_vals[logits.argmax(dim=-1)].long()     # (B,)
            self.last_t_e = te_b.detach().float()
            return _apply_per_image_te(out_full, te_b), te_b.float()


class EarlyExitMeter:
    """Aggregates ExpertEarlyExit diagnostics over an eval run.

    Call `update(model)` after each forward (early_exit must be on, model in
    eval). Then `report()` prints: the fraction of expert groups that ran only
    timestep 1, per-layer/per-expert mean t_e, and the mean per-group routing
    entropy that drove the decision.
    """
    def __init__(self):
        from collections import defaultdict
        self.te = defaultdict(list)      # (layer_name, expert_id) -> list of t_e across batches
        self.ent = defaultdict(list)     # (layer_name, expert_id) -> normalized group entropy
        self.ent_raw = defaultdict(list) # (layer_name, expert_id) -> raw group entropy [0,1]
        self.pruned = defaultdict(list)  # (layer_name, expert_id) -> 1 if pruned this batch else 0
        self.low_T = {}                  # layer_name -> configured low_T (for "at low_T" reporting)

    @torch.no_grad()
    def update(self, model):
        for name, m in model.named_modules():
            em = getattr(m, 'exit_module', None)
            if em is None or em.last_t_e is None:
                continue
            self.low_T[name] = int(em.low_T)
            te = em.last_t_e.detach().float().cpu()              # (B, E)
            prn = em.last_pruned
            prn = prn.detach().float().cpu() if torch.is_tensor(prn) else None
            E = te.shape[1]
            for e in range(E):
                self.te[(name, e)].extend(te[:, e].tolist())
                if prn is not None:
                    self.pruned[(name, e)].extend(prn[:, e].tolist())
                else:
                    self.pruned[(name, e)].extend([0.0] * te.shape[0])
            if em.last_entropy is not None:
                ent = em.last_entropy.detach().float().cpu()     # (B, E)
                for e in range(ent.shape[1]):
                    self.ent[(name, e)].extend(ent[:, e].tolist())
            if em.last_entropy_raw is not None:
                entr = em.last_entropy_raw.detach().float().cpu()  # (B, E)
                for e in range(entr.shape[1]):
                    self.ent_raw[(name, e)].extend(entr[:, e].tolist())

    def report(self):
        all_te = [t for v in self.te.values() for t in v]
        if not all_te:
            print("[early-exit] no data — is --early-exit on and the model in eval()?")
            return
        low_T_values = sorted(set(self.low_T.values())) if self.low_T else [1]
        head_low_T = low_T_values[0] if len(low_T_values) == 1 else low_T_values
        n_low = sum(t == self.low_T.get(name, 1)
                    for (name, _e), v in self.te.items() for t in v)
        all_pruned = [p for v in self.pruned.values() for p in v]
        print("\n===== Early-exit usage =====")
        print(f"expert groups running at low_T={head_low_T}: {n_low}/{len(all_te)} "
              f"({100.0 * n_low / len(all_te):.1f}%)   |   overall mean t_e = "
              f"{sum(all_te) / len(all_te):.2f}")
        if any(all_pruned):
            print(f"expert groups PRUNED (low entropy): {sum(all_pruned)}/{len(all_pruned)} "
                  f"({100.0 * sum(all_pruned) / len(all_pruned):.1f}%)")
        n_active = sum(1 for v in self.te.values() for t in v if t > 0)
        active_frac = n_active / len(all_te)
        num_slots = len(self.te)
        num_blocks = max(1, len(set(name for (name, _e) in self.te)))
        E_per_block = num_slots // num_blocks
        mean_active_te = (sum(t for v in self.te.values() for t in v if t > 0) / n_active) if n_active else 0.0
        print(f"average active experts (t_e>0): {active_frac * E_per_block:.2f}/{E_per_block} per block "
              f"({active_frac * num_slots:.2f}/{num_slots} total)   |   mean t_e among active = {mean_active_te:.2f}")
        te_int = [int(round(t)) for v in self.te.values() for t in v]
        tot = len(te_int)
        hi = max(te_int) if te_int else 0
        hist = {k: 0 for k in range(0, hi + 1)}
        for t in te_int:
            hist[t] += 1
        dist = "  ".join(f"t_e={k}: {100.0 * hist[k] / tot:.1f}%" for k in range(0, hi + 1))
        print(f"t_e distribution (over all image x expert): {dist}")
        for blk in sorted(set(name for (name, _e) in self.te)):
            bt = [int(round(t)) for (name, _e), v in self.te.items() if name == blk for t in v]
            bh = {k: 0 for k in range(0, hi + 1)}
            for t in bt:
                bh[t] += 1
            bd = "  ".join(f"{k}:{100.0 * bh[k] / len(bt):.0f}%" for k in range(0, hi + 1))
            print(f"  {blk}  t_e dist: {bd}")

        print("\nPer-layer / per-expert  (mean t_e | %% at low_T | %% pruned | norm entropy | raw entropy):")
        for (name, e) in sorted(self.te):
            v = self.te[(name, e)]
            lT = self.low_T.get(name, 1)
            frac_low = 100.0 * sum(t == lT for t in v) / len(v)
            pv = self.pruned.get((name, e), [])
            prune_str = f"{100.0 * sum(pv) / len(pv):.1f}%" if pv else "n/a"
            ev = self.ent.get((name, e), [])
            ent_str = f"{sum(ev) / len(ev):+.2f}" if ev else "n/a"
            er = self.ent_raw.get((name, e), [])
            raw_str = f"{sum(er) / len(er):.3f}" if er else "n/a"
            print(f"  {name}  expert {e}: mean t_e = {sum(v) / len(v):.2f}, "
                  f"t_e={lT} in {frac_low:.1f}%, pruned {prune_str}, "
                  f"norm entropy = {ent_str}, raw entropy = {raw_str}")


class EntropyProbe:
    """Analyze the routing-entropy structure of each MoE layer (read-only).

    For every gate it reports, using raw entropy in [0,1] (entropy of the
    router softmax over experts, averaged over timesteps):
      - whole-image entropy (mean over all tokens),
      - object-token vs background-token entropy (tokens split per image by
        spike activity: object = top quantile firing, background = bottom),
      - per-expert mean entropy and mean activity.

    Use it to check whether low-entropy experts are background experts (low
    activity -> safe to prune) or object experts (high activity -> risky).
    Works whenever the gates run; --early-exit is not required.
    """
    def __init__(self, object_quantile=0.75):
        from collections import defaultdict
        self.q = object_quantile
        self.img = defaultdict(lambda: [0.0, 0])      # name -> [sum, count]
        self.obj = defaultdict(lambda: [0.0, 0])
        self.bg = defaultdict(lambda: [0.0, 0])
        self.exp_ent = defaultdict(lambda: [0.0, 0])  # (name, e) -> [sum, count]
        self.exp_act = defaultdict(lambda: [0.0, 0])

    @torch.no_grad()
    def update(self, model):
        for name, m in model.named_modules():
            gate = getattr(m, 'gate', None)
            if gate is None:
                continue
            ginput = getattr(gate, 'last_gate_input', None)    # (T, B, C, H, W) router input
            spikes = getattr(gate, 'last_gate_spikes', None)   # (T, B, C, H, W)
            idx = getattr(gate, 'last_indices', None)
            if ginput is None or spikes is None or idx is None:
                continue
            E = gate.num_gates
            C = ginput.shape[2]
            feat = ginput.mean(dim=0).flatten(2)               # (B, C, N) router input over timesteps
            p = feat.softmax(dim=1)                            # (B, C, N) per-token channel distribution
            ent = -(p * p.clamp_min(1e-9).log()).sum(dim=1) / math.log(C)  # (B, N) in [0,1]
            T_ = spikes.shape[0]
            fir = spikes.flatten(3).sum(dim=(0, 2)).float() / T_   # (B, N) bits/timestep per token

            q_hi = torch.quantile(fir, self.q, dim=1, keepdim=True)
            q_lo = torch.quantile(fir, 1.0 - self.q, dim=1, keepdim=True)
            obj = (fir >= q_hi).float()
            bg = (fir <= q_lo).float()

            def masked_mean(mask):
                c = mask.sum(dim=1).clamp(min=1.0)
                return ((ent * mask).sum(dim=1) / c).mean().item()

            self.img[name][0] += ent.mean().item(); self.img[name][1] += 1
            self.obj[name][0] += masked_mean(obj); self.obj[name][1] += 1
            self.bg[name][0] += masked_mean(bg); self.bg[name][1] += 1

            index = idx[0]                                     # (B, N) top-1 expert per token
            member = F.one_hot(index, E).to(ent.dtype)
            counts = member.sum(dim=(0, 1)).clamp(min=1.0)
            ent_e = torch.einsum('bn,bne->e', ent, member) / counts
            act_e = torch.einsum('bn,bne->e', fir, member) / counts
            for e in range(E):
                self.exp_ent[(name, e)][0] += ent_e[e].item(); self.exp_ent[(name, e)][1] += 1
                self.exp_act[(name, e)][0] += act_e[e].item(); self.exp_act[(name, e)][1] += 1

    def report(self):
        def avg(d, k):
            s, c = d[k]
            return s / c if c else float('nan')
        names = sorted(self.img)
        if not names:
            print("[entropy-probe] no data — did the gates run?")
            return
        print("\n===== Routing-entropy structure (raw entropy in [0,1]) =====")
        print(f"(object = top {int(self.q * 100)}% firing tokens, "
              f"background = bottom {int((1 - self.q) * 100)}%)")
        for name in names:
            ie, oe, be = avg(self.img, name), avg(self.obj, name), avg(self.bg, name)
            print(f"  {name}: image={ie:.3f} | object={oe:.3f} | background={be:.3f} "
                  f"| obj-bg gap={oe - be:+.3f}")
        print("\nPer-expert (mean entropy | mean activity):")
        for (name, e) in sorted(self.exp_ent):
            print(f"  {name} expert {e}: entropy={avg(self.exp_ent, (name, e)):.3f}, "
                  f"activity={avg(self.exp_act, (name, e)):.2f}")


def _apply_per_image_te(out_full, te_b):
    """Apply a PER-IMAGE timestep budget to an expert output by repeat-last.

    out_full: (T, B, ...) expert output run for the full T timesteps.
    te_b:     (B,) long, each image's timestep budget for this expert.
    For image b, timesteps beyond te_b[b] reuse the output at timestep te_b[b]-1
    (the same "run t_e then repeat the last output" rule, but per image). The
    expert is run full-T over the batch — at batch>1 you can't skip timesteps for
    only some images — so this preserves accuracy/decisions exactly while the
    reported mean t_e is the per-sample compute (realized at batch=1 deployment).
    """
    T = out_full.shape[0]
    t_idx = torch.arange(T, device=out_full.device).view(T, 1)            # (T, 1)
    clamp = torch.minimum(t_idx, (te_b - 1).clamp(min=0).view(1, -1))     # (T, B)
    shape = [T, out_full.shape[1]] + [1] * (out_full.dim() - 2)
    idx = clamp.view(*shape).expand_as(out_full)
    return torch.gather(out_full, 0, idx)


def _freeze_at(out_full, te):
    """Repeat the te-th timestep of an expert output for all later steps (scalar te).

    Differentiable freeze-tail used by `ExpertTimestepPredictor` during training:
    out_full (T, B, ...) -> timesteps >= te reuse the output at timestep te-1. Same
    repeat-last rule as `_apply_per_image_te`, but with a single budget for the whole
    batch (the per-budget term inside the expected-output sum).
    """
    T = out_full.shape[0]
    if te >= T:
        return out_full
    tail = out_full[te - 1:te].expand(T - te, *([-1] * (out_full.dim() - 1)))
    return torch.cat([out_full[:te], tail], dim=0)


ENTROPY_CAPTURE = None


@torch.no_grad()
def _output_entropy_te(out_full, theta, min_T, capture_key=None, token_count=None):
    """DT-SNN-style per-expert exit from the entropy of the ACCUMULATED OUTPUT.

    out_full: (T, B, D, 1, Ccap) expert output for the full T timesteps.
    Mirrors DT-SNN/SEENN-I but per expert: after each timestep t we accumulate the
    expert's output, softmax it over channels to a distribution, and take its
    (normalized) entropy. Low entropy = the expert's output has "settled" / is
    confident -> stop. The per-image budget is the first t (>= min_T) whose
    accumulated-output entropy falls below `theta`; if none, run full T.

    Returns a per-IMAGE budget t_e of shape (B,) — computed from each image's own
    output only, so it is invariant to the eval batch size. Higher theta -> easier
    to stop -> fewer timesteps; lower theta -> run longer. No learnable parameters.
    """
    T, B, D = out_full.shape[0], out_full.shape[1], out_full.shape[2]
    min_T = max(1, min(int(min_T), T))
    if token_count is not None:
        cnt = token_count.to(out_full.dtype).clamp(min=1).view(1, B, 1)
    else:
        cnt = (out_full.abs().sum(dim=(0, 2, 3)) > 0).sum(dim=1).clamp(min=1).view(1, B, 1).to(out_full.dtype)
    o = out_full.sum(dim=(3, 4)) / cnt                                     # (T, B, D)
    acc = o.cumsum(dim=0) / torch.arange(1, T + 1, device=o.device, dtype=o.dtype).view(T, 1, 1)
    p = acc.softmax(dim=2)                                          # (T, B, D)
    ent = -(p * p.clamp_min(1e-9).log()).sum(dim=2) / math.log(D)   # (T, B) in [0, 1]
    if ENTROPY_CAPTURE is not None and capture_key is not None:
        ENTROPY_CAPTURE.setdefault(capture_key, []).append(ent.detach().float().cpu())
    confident = ent < theta                                        # (T, B)
    if min_T > 1:
        confident[:min_T - 1] = False                              # must run >= min_T steps
    idx = torch.arange(T, device=o.device).view(T, 1).expand(T, B)
    masked = torch.where(confident, idx, torch.full_like(idx, T))  # T where not confident
    first = masked.min(dim=0).values                               # (B,) first confident t-index, else T
    t_e = (first + 1).clamp(max=T)                                 # #steps run (none confident -> T)
    return t_e.long()


@torch.no_grad()
def _input_entropy_tmean(src, token_count=None, capture_key=None):
    """Single per-image entropy of the expert input, averaged over BOTH tokens AND all
    T timesteps (no cumulative/first-step dependence). Used for pure prune-or-keep:
    prune iff this entropy < threshold. src: (T,B,D,1,Ccap)."""
    T, B, D = src.shape[0], src.shape[1], src.shape[2]
    if token_count is not None:
        cnt = token_count.to(src.dtype).clamp(min=1).view(1, B, 1)
    else:
        cnt = (src.abs().sum(dim=(0, 2, 3)) > 0).sum(dim=1).clamp(min=1).view(1, B, 1).to(src.dtype)
    o = src.sum(dim=(3, 4)) / cnt                                  # (T, B, D) per-timestep token-mean
    o = o.mean(dim=0)                                              # (B, D) average over ALL T timesteps
    p = o.softmax(dim=1)                                           # (B, D) channel distribution
    ent = -(p * p.clamp_min(1e-9).log()).sum(dim=1) / math.log(D)  # (B,) in [0,1]
    if ENTROPY_CAPTURE is not None and capture_key is not None:
        ENTROPY_CAPTURE.setdefault(capture_key, []).append(ent.detach().float().cpu().unsqueeze(0))
    return ent


@torch.no_grad()
def _input_spike_entropy_tmean(src, token_count=None, capture_key=None, base2=False,
                               concentrate="linear", temp=None):
    """Softmax-free SHAPE metric: entropy of the per-CHANNEL spike distribution of the expert input.
    Bins spikes by channel (summed over T and the assigned token slots) into integer counts n_c,
    forms a distribution p_c over channels, and takes its entropy. Same direction/range as
    _input_entropy_tmean (in [0,1], LOW => concentrated / settled => prunable).
      concentrate='linear': p_c = n_c / N  (plain histogram; cannot concentrate -> tends to
                            saturate near max entropy because spike counts are near-uniform).
      concentrate='exp2'  : SOFTERMAX over the counts, p_c = 2^{(n_c-max)/temp} / sum, which DOES
                            concentrate. With temp=1 the exponent n_c-max is a non-positive INTEGER,
                            so 2^{...} is an EXACT bit-shift (no PWL/mantissa, no exp) -- spike-based
                            Softermax for free. temp>1 softens, <1 sharpens.
      base2=False: entropy via natural log (reference).  base2=True: log2 ~ MSB position (priority
                   encoder), the hardware form -- pairs with concentrate='exp2' for a shift+MSB metric.
    src: (T,B,D,1,Ccap) binary spikes; returns (B,) in [0,1]. token_count accepted for parity."""
    T, B, D = src.shape[0], src.shape[1], src.shape[2]
    n = src.sum(dim=(0, 3, 4))                                     # (B, D) integer spike count/channel
    if concentrate == "exp2":
        tau = 1.0 if temp is None else float(temp)
        e = (n - n.max(dim=1, keepdim=True).values) / tau         # <=0; INTEGER when tau=1 (=> exact shift)
        g = torch.exp2(e)                                         # 2^e (right-shifts: 1, 1/2, 1/4, ...)
        Z = g.sum(dim=1).clamp(min=1e-9)                          # (B,) accumulate
        if base2:
            log2Z = torch.floor(torch.log2(Z))                   # MSB(Z) approximation
            H = log2Z - (g * e).sum(dim=1) / Z                    # (B,) bits
            ent = (H / math.log2(D)).clamp(0.0, 1.0)
        else:
            p = g / Z.unsqueeze(1)                                # (B, D) Softermax over counts
            ent = -(p * p.clamp_min(1e-9).log()).sum(dim=1) / math.log(D)
            ent = ent.clamp(0.0, 1.0)
    else:
        N = n.sum(dim=1).clamp(min=1.0)                            # (B,) total spikes
        if base2:
            log2n = torch.where(n >= 1, torch.floor(torch.log2(n.clamp(min=1.0))), torch.zeros_like(n))
            H = torch.floor(torch.log2(N)) - (n * log2n).sum(dim=1) / N
            ent = (H / math.log2(D)).clamp(0.0, 1.0)
        else:
            p = n / N.unsqueeze(1)
            ent = -(p * p.clamp_min(1e-9).log()).sum(dim=1) / math.log(D)
    if ENTROPY_CAPTURE is not None and capture_key is not None:
        ENTROPY_CAPTURE.setdefault(capture_key, []).append(ent.detach().float().cpu().unsqueeze(0))
    return ent


def _pwl_exp2(z):
    """Hardware-cheap piecewise-linear approximation of 2^z for z<=0 (Schraudolph-style):
    2^z = 2^floor(z) * 2^frac(z), with 2^frac ~ 1+frac (linear mantissa). 2^floor(z) is a bit
    shift; the mantissa is one linear interp -- no LUT, no transcendental."""
    fl = torch.floor(z)
    frac = z - fl
    return torch.exp2(fl) * (1.0 + frac)


@torch.no_grad()
def _input_approxsoftmax_entropy_tmean(src, token_count=None, capture_key=None,
                                       base="exp2", temp=None, delta=None, kmax=None):
    """Entropy of an APPROXIMATED softmax of the expert input drive. The point: a count/linear
    distribution is too flat to be discriminative (it can't concentrate); softmax's exp is what
    spreads it. We reproduce that spread with a CHEAP CONVEX surrogate for exp instead of the
    natural exp, then take a base-2 entropy. Because 2^z = e^{z ln2}, base-2 softmax at temp=ln2
    matches the true (base-e) softmax exactly -- so this recovers entropy's discriminability at
    bit-shift cost.
      base='softmax' : exact e^z softmax (reference == _input_entropy_tmean, base-e log).
      base='exp2'    : 2^{z/temp} softmax (exact base-2 exp), base-2 log.
      base='pwl'     : 2^{z/temp} with the piecewise-linear _pwl_exp2 surrogate, base-2 log.
      base='dyadic'  : DYADIC SOFTMAX entropy -- quantize the base-2 exponent to integers so each
                       prob is a power of two (2^{-k_c} = exact right-shift) and the per-channel
                       surprisal is the integer bit-gap k_c itself (the log is FREE). The entropy
                       then closes to  H = (sum_c k_c 2^{-k_c})/Z + log2 Z  with the single log2(Z)
                       via MSB + 1 linear mantissa bit (bit-trick). ZERO exp, ZERO per-channel log.
    temp defaults to ln2 for exp2/pwl/dyadic (=> matches base-e softmax), 1.0 for 'softmax'.
    src: (T,B,D,1,Ccap); returns (B,) in [0,1]."""
    T, B, D = src.shape[0], src.shape[1], src.shape[2]
    if token_count is not None:
        cnt = token_count.to(src.dtype).clamp(min=1).view(1, B, 1)
    else:
        cnt = (src.abs().sum(dim=(0, 2, 3)) > 0).sum(dim=1).clamp(min=1).view(1, B, 1).to(src.dtype)
    o = src.sum(dim=(3, 4)) / cnt                                  # (T, B, D) token-mean
    z = o.mean(dim=0)                                              # (B, D) T-mean drive
    if base == "softmax":
        p = z.softmax(dim=1)
        ent = -(p * p.clamp_min(1e-9).log()).sum(dim=1) / math.log(D)
    elif base == "dyadic":
        tau = float(temp) if temp is not None else math.log(2.0)
        zt = (z - z.max(dim=1, keepdim=True).values) / tau        # <= 0 (softmax-invariant shift)
        k = torch.round(-zt).clamp(min=0.0)                       # integer bit-gap from the peak
        w = torch.exp2(-k)                                        # 2^{-k}: exact right-shift
        if kmax is not None:                                      # DISCARD gaps > kmax (weight 0,
            w = w * (k <= float(kmax)).to(w.dtype)               # not lumped) -- best at kmax~5
        Z = w.sum(dim=1).clamp(min=1.0)                          # >=1 (peak channel contributes 1)
        S = (k * w).sum(dim=1)                                    # sum_c k_c 2^{-k_c}
        e = torch.floor(torch.log2(Z))                          # MSB(Z) (priority encoder)
        log2Z = e + (Z / torch.exp2(e) - 1.0)                    # + 1 linear mantissa bit (bit-trick)
        ent = ((S / Z + log2Z) / math.log2(D)).clamp(0.0, 1.0)
    elif base == "peakmean":
        tau = float(temp) if temp is not None else 1.0
        kref = float(delta) if delta is not None else 8.0
        pm = (z.max(dim=1).values - z.mean(dim=1)) / tau         # (B,)
        ent = (1.0 - pm / kref).clamp(0.0, 1.0)                   # high gap => low => prune
    elif base in ("zonly", "meangap", "hardcount"):
        tau = float(temp) if temp is not None else 1.0
        g = (z.max(dim=1, keepdim=True).values - z) / tau        # >= 0 gaps
        if base == "hardcount":
            dlt = float(delta) if delta is not None else 2.0
            ent = ((g < dlt).float().sum(dim=1) / float(D)).clamp(0.0, 1.0)
        else:
            k = g.round().clamp(min=0.0)
            if base == "zonly":
                w = torch.exp2(-k)
                if kmax is not None:                              # discard gaps > kmax (weight 0)
                    w = w * (k <= float(kmax)).to(w.dtype)
                Z = w.sum(dim=1).clamp(min=1.0)
                ent = (torch.log2(Z) / math.log2(D)).clamp(0.0, 1.0)
            else:  # meangap
                kref = float(delta) if delta is not None else 8.0
                ent = (1.0 - k.mean(dim=1) / kref).clamp(0.0, 1.0)  # high mean-gap => low => prune
    else:
        tau = float(temp) if temp is not None else math.log(2.0)
        z = (z - z.max(dim=1, keepdim=True).values) / tau         # shift for stability (softmax-invariant)
        g = torch.exp2(z) if base == "exp2" else _pwl_exp2(z)     # convex surrogate for exp
        p = g / g.sum(dim=1, keepdim=True)
        ent = -(p * p.clamp_min(1e-9).log2()).sum(dim=1) / math.log2(D)
    ent = ent.clamp(0.0, 1.0)
    if ENTROPY_CAPTURE is not None and capture_key is not None:
        ENTROPY_CAPTURE.setdefault(capture_key, []).append(ent.detach().float().cpu().unsqueeze(0))
    return ent


def _input_spikerate_tmean(src, token_count=None, capture_key=None):
    """Per-image mean INPUT SPIKE RATE of the expert input -- a cheap, softmax-free
    activity metric for prune-or-keep: prune iff the mean firing rate < threshold
    (a near-silent expert input carries little information). Same convention as
    _input_entropy_tmean (lower => more prunable) but only needs a sum, no softmax/log.
    src: (T,B,D,1,Ccap); returns (B,) ~in [0,1] for binary spikes."""
    T, B, D = src.shape[0], src.shape[1], src.shape[2]
    if token_count is not None:
        cnt = token_count.to(src.dtype).clamp(min=1)                       # (B,) tokens routed here
    else:
        cnt = (src.abs().sum(dim=(0, 2, 3)) > 0).sum(dim=1).clamp(min=1).to(src.dtype)
    total = src.sum(dim=(0, 2, 3, 4))                                      # (B,) total spikes over T,D,slots
    rate = total / (cnt * float(D) * float(T))                            # (B,) mean spikes/(token,ch,step)
    if ENTROPY_CAPTURE is not None and capture_key is not None:
        ENTROPY_CAPTURE.setdefault(capture_key, []).append(rate.detach().float().cpu().unsqueeze(0))
    return rate


def _input_entropy_per_t(src, token_count=None, reduction="max", capture_key=None):
    """DVS-appropriate per-image entropy of the expert input.

    Unlike _input_entropy_tmean (which averages over T *before* the softmax and so
    destroys temporal structure and inflates entropy on sparse event data), this
    computes the channel-distribution entropy PER TIMESTEP and only then reduces
    over time -- masking out empty (no-event) timesteps so a blank frame can't read
    as max entropy, and weighting the rest by their event activity. src: (T,B,D,1,Ccap).

    reduction:
      'mean' -> event-activity-weighted mean over non-empty timesteps.
      'max'  -> worst-case (least prunable) non-empty timestep; prune only if every
                timestep is low-entropy.
    Convention matches _input_entropy_tmean: lower value => more prunable
    (prune iff value < threshold). An all-empty image returns 1.0 (keep)."""
    T, B, D = src.shape[0], src.shape[1], src.shape[2]
    if token_count is not None:
        cnt = token_count.to(src.dtype).clamp(min=1).view(1, B, 1)
    else:
        cnt = (src.abs().sum(dim=(0, 2, 3)) > 0).sum(dim=1).clamp(min=1).view(1, B, 1).to(src.dtype)
    o = src.sum(dim=(3, 4)) / cnt                                  # (T,B,D) per-timestep token-mean
    mass = o.abs().sum(dim=2)                                      # (T,B) per-timestep activity
    nonempty = mass > 0                                            # (T,B)
    p = o.softmax(dim=2)                                           # (T,B,D) per-timestep channel dist
    H = -(p * p.clamp_min(1e-9).log()).sum(dim=2) / math.log(D)    # (T,B) per-timestep entropy in [0,1]
    if reduction == "max":
        Hm = H.masked_fill(~nonempty, float('-inf'))
        ent, _ = Hm.max(dim=0)                                     # (B,)
        ent = torch.where(nonempty.any(dim=0), ent, torch.ones_like(ent))
    else:  # event-activity-weighted mean over non-empty timesteps
        w = mass * nonempty.to(mass.dtype)                        # (T,B)
        denom = w.sum(dim=0)                                      # (B,)
        ent = (H * w).sum(dim=0) / denom.clamp_min(1e-9)         # (B,)
        ent = torch.where(denom > 0, ent, torch.ones_like(ent))
    if ENTROPY_CAPTURE is not None and capture_key is not None:
        ENTROPY_CAPTURE.setdefault(capture_key, []).append(ent.detach().float().cpu().unsqueeze(0))
    return ent


@torch.no_grad()
def _output_convergence_te(out_full, eps, min_T):
    """Per-expert exit when the ACCUMULATED OUTPUT stops changing between timesteps.

    out_full: (T, B, D, 1, Ccap). Complements `_output_entropy_te`: an expert can have a
    high-entropy output that is nonetheless TEMPORALLY STABLE (the running-mean output
    barely changes from step t-1 to t) — entropy never lets it exit, but convergence does.
    This is what can cut block-0 experts (uniformly high output-entropy, uncuttable by
    entropy) when their output has converged.

    Relative change at step t: ||acc_t - acc_{t-1}|| / (||acc_t|| + 1e-6), over channels.
    Exit at the first t (>= max(min_T, 2)) whose relative change < eps. Higher eps -> easier
    to stop -> fewer timesteps. Per-IMAGE, batch-invariant, no learnable parameters.
    """
    T, B, D = out_full.shape[0], out_full.shape[1], out_full.shape[2]
    min_T = max(1, min(int(min_T), T))
    o = out_full.mean(dim=(3, 4))                                   # (T, B, D)
    acc = o.cumsum(dim=0) / torch.arange(1, T + 1, device=o.device, dtype=o.dtype).view(T, 1, 1)
    diff = acc[1:] - acc[:-1]                                       # (T-1, B, D) change at steps 2..T
    rel = diff.norm(dim=2) / (acc[1:].norm(dim=2) + 1e-6)           # (T-1, B)
    rel = torch.cat([rel.new_full((1, B), float('inf')), rel], 0)  # (T, B); step 1 can't be "converged"
    confident = rel < eps                                          # (T, B)
    lo = max(min_T, 2)                                             # need >= 2 steps to measure a change
    if lo > 1:
        confident[:lo - 1] = False
    idx = torch.arange(T, device=o.device).view(T, 1).expand(T, B)
    masked = torch.where(confident, idx, torch.full_like(idx, T))
    first = masked.min(dim=0).values
    t_e = (first + 1).clamp(max=T)
    return t_e.long()


class MoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 4,
        hidden_features=None,
        out_features=None,
        spike_mode="lif",
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 4.,
        capacity_factor_eval = 4.,
        loss_coef = 1e-2,

        top_k = 1,
        mixing_mode = 'post_linear',
        mix_ratio = 0.1,
        sample_routing = False,
        use_ste = False,
        use_output_lif = False,
        early_exit = False,
        exit_threshold = 0.5,
        exit_low_T = 1,
        prune_threshold = None,
        entropy_norm = True,
        exit_metric = 'entropy',
        exit_mode = 'absolute',
        experts = None):
        super().__init__()

        self.num_experts = num_experts

        gating_kwargs = {'top_k' : top_k, 'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval, 'sample_routing': sample_routing, 'use_ste': use_ste, 'early_exit': early_exit, 'exit_threshold': exit_threshold, 'exit_low_T': exit_low_T}
        self.gate = Top2Gating(dim, num_gates = num_experts - 1, mixing_mode=mixing_mode, mix_ratio=mix_ratio, **gating_kwargs)

        
        self.experts = nn.ModuleList([
            MS_MLP_Expert(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=out_features,
                spike_mode='lif',
                tau=2.0,
                use_output_lif=use_output_lif,

            )
            for i in range(num_experts)
        ])

        self.early_exit = early_exit
        self.exit_module = ExpertEarlyExit(num_experts - 1, entropy_threshold=exit_threshold,
                                           low_T=exit_low_T, prune_threshold=prune_threshold,
                                           per_layer_norm=entropy_norm, metric=exit_metric,
                                           exit_mode=exit_mode)
        self.expert_timesteps = None
        self.only_expert_ids = None

        self.loss_coef = loss_coef

    def forward(self, inputs, hook = None, **kwargs):
        T, B, D, H, W = inputs.shape
        N = H * W
        E = self.num_experts
        R = E - 1  # number of routed experts (== gate num_gates)

        fixed_output, _ = self.experts[0](inputs, hook=hook)  # (T, B, D, H, W)

        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        Ccap = combine_tensor.shape[-1]

        if self.only_expert_ids is not None:
            keep = torch.zeros((1, 1, R, 1), device=combine_tensor.device, dtype=combine_tensor.dtype)
            keep_ids = [eid for eid in self.only_expert_ids if 0 <= eid < R]
            if keep_ids:
                keep[:, :, keep_ids, :] = 1.0
            dispatch_tensor = dispatch_tensor * keep
            combine_tensor = combine_tensor * keep
            selected = set(keep_ids)
        else:
            selected = None

        te_per_image = None   # (B, R) when early-exit on
        prune_per_image = None
        if self.early_exit and not self.training:
            te_per_image = self.exit_module(self.gate.last_gate_input, self.gate.last_gate_spikes, self.gate.last_indices[0])
            prune_per_image = self.exit_module.last_pruned
        elif self.expert_timesteps is not None:
            t_e_list = self.expert_timesteps
        else:
            t_e_list = [T] * R

        x_tok = inputs.flatten(3).permute(0, 1, 3, 2).contiguous()  # T, B, N, D
        expert_inputs = torch.einsum('tbnd,bnec->tebcd', x_tok, dispatch_tensor.to(x_tok.dtype))

        expert_outputs_list = []
        for i in range(R):
            if selected is not None and i not in selected:
                expert_outputs_list.append(
                    torch.zeros((T, B, Ccap, D), device=inputs.device, dtype=inputs.dtype)
                )
                continue
            expert_input = expert_inputs[:, i, :, :, :]  # (T, B, Ccap, D)
            expert_input_spatial = expert_input.permute(0, 1, 3, 2).unsqueeze(-2)  # T, B, D, 1, Ccap

            if te_per_image is not None:
                out_full, _ = self.experts[i + 1](expert_input_spatial, hook=hook)
                expert_output_spatial = _apply_per_image_te(out_full, te_per_image[:, i])
                if prune_per_image is not None:
                    keep = (~prune_per_image[:, i]).view(1, B, 1, 1, 1).to(expert_output_spatial.dtype)
                    expert_output_spatial = expert_output_spatial * keep
            else:
                t_e = min(t_e_list[i], T)
                expert_output_trimmed, _ = self.experts[i + 1](expert_input_spatial[:t_e], hook=hook)
                if t_e < T:
                    expert_output_spatial = torch.cat(
                        [expert_output_trimmed,
                         expert_output_trimmed[-1:].expand(T - t_e, -1, -1, -1, -1)],
                        dim=0,
                    )
                else:
                    expert_output_spatial = expert_output_trimmed

            expert_output = expert_output_spatial.squeeze(-2).permute(0, 1, 3, 2)  # (T, B, Ccap, D)
            expert_outputs_list.append(expert_output)

        expert_outputs = torch.stack(expert_outputs_list, dim=1)  # (T, R, B, Ccap, D)
        routed_output = torch.einsum('tebcd,bnec->tbnd', expert_outputs, combine_tensor)  # (T, B, N, D)
        routed_output = routed_output.permute(0, 1, 3, 2).reshape(T, B, D, H, W)

        output = fixed_output + routed_output + inputs

        return output, loss * self.loss_coef, hook


class MoEAllRouted(nn.Module):
    """MoE variant with no fixed expert: top-k selected from all E experts."""
    def __init__(self,
        dim,
        num_experts=4,
        hidden_features=None,
        out_features=None,
        spike_mode="lif",
        second_policy_train='random',
        second_policy_eval='random',
        second_threshold_train=0.2,
        second_threshold_eval=0.2,
        capacity_factor_train=4.,
        capacity_factor_eval=4.,
        loss_coef=1e-2,
        top_k=1,
        mixing_mode='post_linear',
        mix_ratio=0.1,
        sample_routing=False,
        use_ste=False,
        use_output_lif=False,
        early_exit=False,
        exit_threshold=0.5,
        exit_low_T=1,
        prune_threshold=None,
        entropy_norm=True,
        exit_metric='entropy',
        exit_mode='absolute',
        halting=False,
        halt_lambda=1e-2,
        halt_min_T=1,
        trunc_train=False,
        trunc_min_T=1,
        T=4,
        layer=0,
        experts=None):
        super().__init__()

        self.num_experts = num_experts

        gating_kwargs = {
            'top_k': top_k,
            'second_policy_train': second_policy_train,
            'second_policy_eval': second_policy_eval,
            'second_threshold_train': second_threshold_train,
            'second_threshold_eval': second_threshold_eval,
            'capacity_factor_train': capacity_factor_train,
            'capacity_factor_eval': capacity_factor_eval,
            'sample_routing': sample_routing,
            'use_ste': use_ste,
            'early_exit': early_exit,
            'exit_threshold': exit_threshold,
            'exit_low_T': exit_low_T,
        }
        self.gate = Top2Gating(dim, num_gates=num_experts, mixing_mode=mixing_mode, mix_ratio=mix_ratio, **gating_kwargs)

        self.experts = nn.ModuleList([
            MS_MLP_Expert(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=out_features,
                spike_mode='lif',
                tau=2.0,
                use_output_lif=use_output_lif,
            )
            for _ in range(num_experts)
        ])

        self.early_exit = early_exit
        self.exit_metric = exit_metric
        self.exit_threshold = exit_threshold
        self.exit_low_T = exit_low_T
        _OE_METRICS = ('output_entropy', 'output_convergence', 'input_entropy', 'input_convergence')
        _ee_metric = 'entropy' if exit_metric in _OE_METRICS else exit_metric
        self.exit_module = ExpertEarlyExit(num_experts, entropy_threshold=exit_threshold,
                                           low_T=exit_low_T, prune_threshold=prune_threshold,
                                           per_layer_norm=entropy_norm, metric=_ee_metric,
                                           exit_mode=exit_mode)
        self.expert_timesteps = None
        self.only_expert_ids = None
        self.prune_perimage_k = None

        self.halting = halting
        self.halt_lambda = halt_lambda
        self.layer = layer
        if halting:
            self.ts_predictors = nn.ModuleList([
                ExpertTimestepPredictor(dim, T=T, min_T=halt_min_T)
                for _ in range(num_experts)
            ])
        self.last_halt_te = None   # (B, E) per-image budget when halting on (eval)

        self.trunc_train = trunc_train
        self.trunc_min_T = trunc_min_T

        self.loss_coef = loss_coef

    def forward(self, inputs, hook=None, **kwargs):
        T, B, D, H, W = inputs.shape
        E = self.num_experts

        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        Ccap = combine_tensor.shape[-1]

        if self.only_expert_ids is not None:
            keep = torch.zeros((1, 1, E, 1), device=combine_tensor.device, dtype=combine_tensor.dtype)
            keep_ids = [eid for eid in self.only_expert_ids if 0 <= eid < E]
            if keep_ids:
                keep[:, :, keep_ids, :] = 1.0
            dispatch_tensor = dispatch_tensor * keep
            combine_tensor = combine_tensor * keep
            selected = set(keep_ids)
        else:
            selected = None

        te_per_image = None   # (B, E) when input-side early-exit on
        prune_per_image = None
        prune_active = (not self.training) or getattr(self, 'prune_in_train', False)
        oe_mode = self.early_exit and prune_active and (self.exit_metric in
                  ('output_entropy', 'output_convergence', 'input_entropy', 'input_convergence'))
        oe_te_cols = []       # per-expert (B,) output-entropy budgets, for the meter
        if self.early_exit and prune_active and not oe_mode:
            te_per_image = self.exit_module(self.gate.last_gate_input, self.gate.last_gate_spikes, self.gate.last_indices[0])
            prune_per_image = self.exit_module.last_pruned
        elif self.expert_timesteps is not None:
            t_e_list = self.expert_timesteps
        else:
            t_e_list = [T] * E

        trunc_mode = self.trunc_train and self.training
        te_rand = None
        if trunc_mode:
            lo = max(0, min(self.trunc_min_T, T))   # 0 -> includes pruning (expert dropout)
            te_rand = torch.randint(lo, T + 1, (B, E), device=inputs.device)  # (B, E)

        prune_pi_mask = None   # (B, E) bool, True = prune this expert for this image
        pi_mode = (self.prune_perimage_k is not None) and prune_active
        if pi_mode:
            K = int(self.prune_perimage_k)
            load = dispatch_tensor.to(inputs.dtype).sum(dim=(1, 3))       # (B, E) tokens/expert/image
            low = load.topk(min(K, E), dim=1, largest=False).indices      # (B, K) lowest-load experts
            prune_pi_mask = torch.zeros((B, E), dtype=torch.bool, device=inputs.device)
            prune_pi_mask.scatter_(1, low, True)

        x_tok = inputs.flatten(3).permute(0, 1, 3, 2).contiguous()  # T, B, N, D
        expert_inputs = torch.einsum('tbnd,bnec->tebcd', x_tok, dispatch_tensor.to(x_tok.dtype))
        expert_load = dispatch_tensor.to(inputs.dtype).sum(dim=(1, 3))    # (B, E)
        self.last_expert_load = expert_load.detach()                     # for analysis (tokens/expert/image)

        gate_spk_inputs = None
        if getattr(self, 'prune_spikecount', False) or getattr(self, 'prune_spike_entropy', False):
            gs = getattr(self.gate, 'last_gate_spikes', None)
            if gs is not None and gs.flatten(3).shape[:2] == x_tok.shape[:2] and gs.shape[2] == D:
                gs_tok = gs.flatten(3).permute(0, 1, 3, 2).contiguous()   # T, B, N, D
                gate_spk_inputs = torch.einsum('tbnd,bnec->tebcd', gs_tok,
                                               dispatch_tensor.to(gs_tok.dtype))

        ponder_terms = []     # per-expert expected #steps (B,) when halting on
        halt_te_cols = []     # per-expert per-image budget for diagnostics
        expert_outputs_list = []
        for i in range(E):
            if selected is not None and i not in selected:
                expert_outputs_list.append(
                    torch.zeros((T, B, Ccap, D), device=inputs.device, dtype=inputs.dtype)
                )
                continue
            expert_input = expert_inputs[:, i, :, :, :]          # T, B, Ccap, D
            expert_input_spatial = expert_input.permute(0, 1, 3, 2).unsqueeze(-2)  # T, B, D, 1, Ccap
            gate_spk_spatial = (gate_spk_inputs[:, i].permute(0, 1, 3, 2).unsqueeze(-2)
                                if gate_spk_inputs is not None else None)  # T, B, D, 1, Ccap (true spikes)

            if pi_mode:
                out_full, _ = self.experts[i](expert_input_spatial, hook=hook)
                keep = (~prune_pi_mask[:, i]).view(1, B, 1, 1, 1).to(out_full.dtype)
                expert_output_spatial = out_full * keep
            elif self.halting:
                out_full, _ = self.experts[i](expert_input_spatial, hook=hook)
                expert_output_spatial, te_or_ponder = self.ts_predictors[i](expert_input_spatial, out_full)
                if self.training:
                    ponder_terms.append(te_or_ponder)            # (B,) expected steps
                else:
                    halt_te_cols.append(te_or_ponder)            # (B,) hard budget
            elif trunc_mode:
                out_full, _ = self.experts[i](expert_input_spatial, hook=hook)
                expert_output_spatial = _apply_per_image_te(out_full, te_rand[:, i].clamp(min=1))
                if self.trunc_min_T == 0:
                    keep = (te_rand[:, i] > 0).view(1, B, 1, 1, 1).to(expert_output_spatial.dtype)
                    expert_output_spatial = expert_output_spatial * keep
            elif oe_mode:
                out_full, _ = self.experts[i](expert_input_spatial, hook=hook)
                src = expert_input_spatial if self.exit_metric.startswith('input') else out_full
                if getattr(self, 'prune_tmean', False) and self.exit_metric.startswith('input'):
                    te_i = torch.full((B,), T, dtype=torch.long, device=inputs.device)
                elif 'convergence' in self.exit_metric:
                    te_i = _output_convergence_te(src, self.exit_threshold, self.exit_low_T)
                else:
                    te_i = _output_entropy_te(src, self.exit_threshold, self.exit_low_T,
                                              capture_key=('te', self.layer, i),
                                              token_count=expert_load[:, i])
                pm = getattr(self, 'exit_prune_metric', None)
                pb = getattr(self, 'exit_prune_below', 0)
                prune_i = None
                if getattr(self, 'prune_tmean', False) and self.exit_metric.startswith('input'):
                    if getattr(self, 'prune_approx_entropy', False):
                        se = _input_approxsoftmax_entropy_tmean(
                            expert_input_spatial, token_count=expert_load[:, i],
                            capture_key=('te', self.layer, i),
                            base=getattr(self, 'approx_base', 'pwl'),
                            temp=getattr(self, 'approx_temp', None),
                            delta=getattr(self, 'approx_delta', None),
                            kmax=getattr(self, 'approx_kmax', None))
                        if getattr(self, 'prune_entropy_high', False):
                            prune_i = se > self.exit_threshold
                        else:
                            prune_i = se < self.exit_threshold
                    elif getattr(self, 'prune_spike_entropy', False):
                        if getattr(self, 'prune_spikecount_source', 'gate') == 'expert_lif':
                            spk_src = getattr(self.experts[i], 'last_fc1_spikes', None)
                            if spk_src is None:
                                spk_src = gate_spk_spatial if gate_spk_spatial is not None else expert_input_spatial
                        else:
                            spk_src = gate_spk_spatial if gate_spk_spatial is not None else expert_input_spatial
                        se = _input_spike_entropy_tmean(
                            spk_src, token_count=expert_load[:, i], capture_key=('te', self.layer, i),
                            base2=getattr(self, 'spike_entropy_base2', False),
                            concentrate=getattr(self, 'spike_entropy_concentrate', 'linear'),
                            temp=getattr(self, 'spike_entropy_temp', None))
                        if getattr(self, 'prune_entropy_high', False):
                            prune_i = se > self.exit_threshold   # HIGH (diffuse) => prune
                        else:
                            prune_i = se < self.exit_threshold   # LOW (concentrated) => prune (default)
                    elif getattr(self, 'prune_spikecount', False):
                        if getattr(self, 'prune_spikecount_source', 'gate') == 'expert_lif':
                            spk_src = getattr(self.experts[i], 'last_fc1_spikes', None)
                            if spk_src is None:
                                spk_src = gate_spk_spatial if gate_spk_spatial is not None else expert_input_spatial
                        else:
                            spk_src = gate_spk_spatial if gate_spk_spatial is not None else expert_input_spatial
                        rate = _input_spikerate_tmean(
                            spk_src, token_count=expert_load[:, i], capture_key=('te', self.layer, i))
                        prune_i = rate > self.exit_threshold
                        if getattr(self, 'prune_zero_spikes', False):
                            prune_i = prune_i | (rate <= getattr(self, 'zero_spike_eps', 1e-6))
                    else:
                        if getattr(self, 'prune_dvs_temporal', False):
                            ent_tm = _input_entropy_per_t(
                                expert_input_spatial, token_count=expert_load[:, i],
                                reduction=getattr(self, 'prune_dvs_reduction', 'max'),
                                capture_key=('te', self.layer, i))
                        else:
                            ent_tm = _input_entropy_tmean(expert_input_spatial, token_count=expert_load[:, i],
                                                          capture_key=('te', self.layer, i))
                        # PER-EXPERT threshold: use this expert's own threshold if set (deployed
                        # sensitivity-allocated point); else the scalar block threshold.
                        _tpe = getattr(self, 'exit_threshold_per_expert', None)
                        _thr = _tpe[i] if _tpe is not None else self.exit_threshold
                        if getattr(self, 'prune_entropy_high', False):
                            prune_i = ent_tm > _thr   # HIGH entropy => prune
                        else:
                            prune_i = ent_tm < _thr   # LOW entropy => prune (default)
                elif pm:
                    psrc = expert_input_spatial if pm.startswith('input') else out_full
                    pth = getattr(self, 'exit_prune_threshold', self.exit_threshold)
                    if 'convergence' in pm:
                        te_p = _output_convergence_te(psrc, pth, self.exit_low_T)
                    else:
                        te_p = _output_entropy_te(psrc, pth, self.exit_low_T,
                                                  capture_key=('prune', self.layer, i),
                                                  token_count=expert_load[:, i])
                    prune_i = te_p < (pb if pb else 2)   # prune metric exits early -> drop expert
                elif pb:
                    prune_i = te_i < pb
                if prune_i is not None and getattr(self, 'prune_only', False):
                    expert_output_spatial = out_full * (~prune_i).view(1, B, 1, 1, 1).to(out_full.dtype)
                    te_i = torch.where(prune_i, torch.zeros_like(te_i), torch.full_like(te_i, T))
                elif prune_i is not None:
                    expert_output_spatial = _apply_per_image_te(out_full, te_i.clamp(min=1))
                    expert_output_spatial = expert_output_spatial * (~prune_i).view(1, B, 1, 1, 1).to(out_full.dtype)
                    te_i = torch.where(prune_i, torch.zeros_like(te_i), te_i)
                else:
                    expert_output_spatial = _apply_per_image_te(out_full, te_i)
                oe_te_cols.append(te_i)
            elif te_per_image is not None:
                out_full, _ = self.experts[i](expert_input_spatial, hook=hook)
                expert_output_spatial = _apply_per_image_te(out_full, te_per_image[:, i])
                if prune_per_image is not None:
                    keep = (~prune_per_image[:, i]).view(1, B, 1, 1, 1).to(expert_output_spatial.dtype)
                    expert_output_spatial = expert_output_spatial * keep
            else:
                t_e = min(t_e_list[i], T)
                if t_e <= 0:
                    expert_output_spatial = torch.zeros(
                        (T, B, D, 1, Ccap), device=inputs.device, dtype=inputs.dtype)
                else:
                    expert_output_trimmed, _ = self.experts[i](expert_input_spatial[:t_e], hook=hook)
                    if t_e < T:
                        expert_output_spatial = torch.cat(
                            [expert_output_trimmed,
                             expert_output_trimmed[-1:].expand(T - t_e, -1, -1, -1, -1)],
                            dim=0,
                        )
                    else:
                        expert_output_spatial = expert_output_trimmed

            expert_output = expert_output_spatial.squeeze(-2).permute(0, 1, 3, 2)  # T, B, Ccap, D
            expert_outputs_list.append(expert_output)

        expert_outputs = torch.stack(expert_outputs_list, dim=1)  # T, E, B, Ccap, D
        routed_output = torch.einsum('tebcd,bnec->tbnd', expert_outputs, combine_tensor)  # (T, B, N, D)
        routed_output = routed_output.permute(0, 1, 3, 2).reshape(T, B, D, H, W)

        output = routed_output + inputs

        if self.halting and self.training and ponder_terms:
            ponder = self.halt_lambda * torch.stack(ponder_terms, dim=1).mean()
            if hook is not None:
                hook[f"ponder_loss_layer_{self.layer}"] = ponder
        if self.halting and not self.training and halt_te_cols:
            self.last_halt_te = torch.stack(halt_te_cols, dim=1)   # (B, E)
        if oe_mode and oe_te_cols:
            te_oe = torch.stack(oe_te_cols, dim=1)          # (B, E)
            self.exit_module.last_t_e = te_oe
            self.exit_module.last_pruned = (te_oe == 0)
            self.exit_module.last_entropy = None
            self.exit_module.last_entropy_raw = None
        if pi_mode:
            te_pi = torch.where(prune_pi_mask, torch.zeros_like(prune_pi_mask, dtype=torch.long),
                                torch.full_like(prune_pi_mask, T, dtype=torch.long))
            self.exit_module.last_t_e = te_pi               # (B, E)
            self.exit_module.last_pruned = prune_pi_mask
            self.exit_module.last_entropy = None
            self.exit_module.last_entropy_raw = None

        return output, loss * self.loss_coef, hook


class PlainMLP(nn.Module):
    """Single MLP (no routing) with MoE-compatible forward signature."""
    def __init__(self, dim, hidden_features, out_features, spike_mode='lif', layer=0, use_output_lif=False):
        super().__init__()
        self.mlp = MS_MLP_Expert(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=out_features,
            spike_mode=spike_mode,
            layer=layer,
            use_output_lif=use_output_lif,
        )

    def forward(self, x, hook=None):
        out, hook = self.mlp(x, hook=hook)
        return out + x, torch.tensor(0.0, device=x.device), hook


class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
        num_experts = 4,
        loss_coef=1e-2,

        top_k = None,
        mixing_mode = 'post_linear',
        mix_ratio = 0.1,
        moe_type = 'moe',
        sample_routing = False,
        use_ste = False,
        use_output_lif = False,
        use_moe = True,
        capacity_factor_train = 4.,
        capacity_factor_eval = 4.,
        early_exit = False,
        exit_threshold = 0.5,
        exit_low_T = 1,
        prune_threshold = None,
        entropy_norm = True,
        exit_metric = 'entropy',
        exit_mode = 'absolute',
        halting = False,
        halt_lambda = 1e-2,
        halt_min_T = 1,
        trunc_train = False,
        trunc_min_T = 1,
        T = 4,
    ):
        super().__init__()

        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            mode=attn_mode,
            spike_mode=spike_mode,
            dvs=dvs,
            layer=layer,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer = layer

        mlp_hidden_dim = int(dim * mlp_ratio)

        if use_moe:
            MoEClass = MoEAllRouted if moe_type == 'allrouted' else MoE
            effective_top_k = top_k if top_k is not None else (2 if moe_type == 'allrouted' else 1)
            moe_kwargs = dict(
                dim=dim,
                num_experts=num_experts,
                hidden_features=mlp_hidden_dim,
                out_features=dim,
                spike_mode='lif',
                loss_coef=loss_coef,
                top_k=effective_top_k,
                mixing_mode=mixing_mode,
                mix_ratio=mix_ratio,
                sample_routing=sample_routing,
                use_ste=use_ste,
                use_output_lif=use_output_lif,
                capacity_factor_train=capacity_factor_train,
                capacity_factor_eval=capacity_factor_eval,
                early_exit=early_exit,
                exit_threshold=exit_threshold,
                exit_low_T=exit_low_T,
                prune_threshold=prune_threshold,
                entropy_norm=entropy_norm,
                exit_metric=exit_metric,
                exit_mode=exit_mode,
            )
            if moe_type == 'allrouted':
                moe_kwargs.update(halting=halting, halt_lambda=halt_lambda,
                                  halt_min_T=halt_min_T, trunc_train=trunc_train,
                                  trunc_min_T=trunc_min_T, T=T, layer=layer)
            self.mlp = MoEClass(**moe_kwargs)
        else:
            self.mlp = PlainMLP(
                dim=dim,
                hidden_features=mlp_hidden_dim,
                out_features=dim,
                spike_mode='lif',
                layer=layer,
                use_output_lif=use_output_lif,
            )

    def forward(self, x, hook=None):
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, moe_loss, hook = self.mlp(x_attn, hook=hook)
        if hook is not None:
            hook[f"moe_loss_layer_{self.layer}"] = moe_loss
        return x, attn, hook

