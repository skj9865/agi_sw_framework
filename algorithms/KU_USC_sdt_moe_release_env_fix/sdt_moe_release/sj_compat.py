"""Compatibility layer so the release runs on an older stack as well as the pinned one.

Pinned env (env/environment.yml): torch 2.7, timm 1.0, spikingjelly 0.0.0.0.15
Older env (e.g. "spikformer"):    torch 1.10, timm 0.5.4, spikingjelly 0.0.0.0.12

Each piece is picked by the installed package VERSION and confirmed by importing the
MODULE (see "version checks" below); the choice is printed by compat_report() at startup.
On the pinned env everything resolves to the real library and this file changes nothing.
On the older env it supplies:

  * LIFNode / ParametricLIFNode with the spikingjelly ``activation_based`` API
    (step_mode, backend, v_float_to_tensor, neuronal_charge/fire/reset, reset()).
    spikingjelly 0.0.0.0.12 only ships the old ``clock_driven`` API.
    At eval time spikingjelly 0.0.0.0.15 ignores ``backend`` and runs a TorchScript loop
    (LIFNode.jit_eval_multi_step_forward_hard_reset_decay_input); the same arithmetic is
    reproduced here so inference matches. The training path uses a plain torch loop with a
    sigmoid surrogate gradient (the cupy kernels are not ported).
  * functional.reset_net
  * trunc_normal_, DropPath, to_2tuple (timm.layers moved from timm.models.layers)
  * clean_state_dict (added to timm after 0.5.4)
  * torch_load: passes weights_only=False only on torch >= 1.13
  * CIFAR10DVS / DVS128Gesture / NCaltech101 whose cached frames load as float32
    (spikingjelly < 0.0.0.0.15 returns float64, which breaks the AMP forward)

Related fix outside this file: dvs_utils.Resize pins antialias=True (torchvision's tensor
default before 0.17 was False, which changes the resized CIFAR10-DVS frames).
"""
import logging
import math
import re
from collections import OrderedDict
from typing import Callable, Optional

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------- version checks
# Each shim is chosen by the INSTALLED VERSION and then confirmed by the MODULE:
#   version >= minimum and module imports  -> use the real library
#   version >= minimum but import fails    -> ImportError (broken/partial install; never
#                                             silently swap in the shim)
#   version <  minimum                     -> use the shim below
#   version unknown                        -> decided by whether the module imports
MIN_TORCH_WEIGHTS_ONLY = (1, 13)          # torch.load(weights_only=...) added in 1.13
MIN_TIMM_LAYERS = (0, 9)                  # timm.layers + timm.models.clean_state_dict
MIN_SJ_ACTIVATION_BASED = (0, 0, 0, 0, 13)  # spikingjelly.activation_based


def _parse_version(v):
    """'2.7.1+cu128' -> (2, 7, 1); '0.0.0.0.12' -> (0, 0, 0, 0, 12); None -> None."""
    if not v:
        return None
    parts = []
    for p in str(v).split("+")[0].split("."):
        m = re.match(r"\d+", p)
        if not m:
            break
        parts.append(int(m.group()))
    return tuple(parts) or None


def _dist_version(dist, module=None):
    try:
        from importlib.metadata import version  # python >= 3.8
        return version(dist)
    except Exception:
        return getattr(module, "__version__", None) if module is not None else None


def _at_least(ver, minimum):
    if ver is None:
        return None
    n = max(len(ver), len(minimum))
    return ver + (0,) * (n - len(ver)) >= minimum + (0,) * (n - len(minimum))


def _choose(name, ver, minimum, try_import):
    """Return (use_real, imported_objects). See the policy above."""
    new_enough = _at_least(ver, minimum)
    if new_enough is False:
        return False, None
    try:
        objs = try_import()
    except ImportError as e:
        if new_enough:
            raise ImportError(
                f"{name} version {'.'.join(map(str, ver))} should provide this API "
                f"(>= {'.'.join(map(str, minimum))}) but the import failed: {e}"
            ) from e
        return False, None
    return True, objs


import timm  # noqa: E402

TORCH_VERSION = _parse_version(torch.__version__)
TIMM_VERSION = _parse_version(_dist_version("timm", timm))
try:
    import spikingjelly as _sj
    SJ_VERSION = _parse_version(_dist_version("spikingjelly", _sj))
except ImportError:
    SJ_VERSION = None

# ---------------------------------------------------------------------------------- torch
TORCH_HAS_WEIGHTS_ONLY = bool(_at_least(TORCH_VERSION, MIN_TORCH_WEIGHTS_ONLY))


def torch_load(path, map_location="cpu"):
    """torch.load with weights_only=False on torch >= 1.13 (where the default flipped to
    True in 2.6); older torch has no such argument and always loads full objects."""
    if TORCH_HAS_WEIGHTS_ONLY:
        return torch.load(path, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------------------- timm
def _import_timm_new():
    from timm.layers import trunc_normal_, DropPath, to_2tuple
    from timm.models import clean_state_dict
    return trunc_normal_, DropPath, to_2tuple, clean_state_dict


USE_TIMM_LAYERS, _objs = _choose("timm", TIMM_VERSION, MIN_TIMM_LAYERS, _import_timm_new)
if USE_TIMM_LAYERS:
    trunc_normal_, DropPath, to_2tuple, clean_state_dict = _objs
else:
    from timm.models.layers import trunc_normal_, DropPath, to_2tuple  # timm < 0.9

    def clean_state_dict(state_dict):
        # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
        cleaned_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            cleaned_state_dict[name] = v
        return cleaned_state_dict


# --------------------------------------------------------------------------- spikingjelly
def _import_sj_new():
    from spikingjelly.activation_based.neuron import LIFNode, ParametricLIFNode
    from spikingjelly.activation_based import functional
    return LIFNode, ParametricLIFNode, functional


HAS_ACTIVATION_BASED, _objs = _choose("spikingjelly", SJ_VERSION, MIN_SJ_ACTIVATION_BASED,
                                      _import_sj_new)
if HAS_ACTIVATION_BASED:
    LIFNode, ParametricLIFNode, functional = _objs
del _objs


def compat_report():
    """One-line summary of detected versions and which implementation is in use."""
    fmt = lambda v: ".".join(map(str, v)) if v else "unknown"  # noqa: E731
    return (f"compat: torch {fmt(TORCH_VERSION)} (weights_only={'yes' if TORCH_HAS_WEIGHTS_ONLY else 'no'}), "
            f"timm {fmt(TIMM_VERSION)} ({'timm.layers' if USE_TIMM_LAYERS else 'shim: timm.models.layers'}), "
            f"spikingjelly {fmt(SJ_VERSION)} "
            f"({'activation_based' if HAS_ACTIVATION_BASED else 'shim: LIF/PLIF/reset_net'}"
            f"{', frames cast to float32' if SJ_FRAMES_NEED_CAST else ''})")


if not HAS_ACTIVATION_BASED:

    class _SigmoidSurrogate(torch.autograd.Function):
        """Heaviside forward, sigmoid'(alpha*x)*alpha backward (spikingjelly surrogate.Sigmoid, alpha=4)."""

        @staticmethod
        def forward(ctx, x, alpha):
            if x.requires_grad:
                ctx.save_for_backward(x)
                ctx.alpha = alpha
            return (x >= 0).to(x)

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            sg = torch.sigmoid(x * ctx.alpha)
            return grad_output * (1.0 - sg) * sg * ctx.alpha, None

    class Sigmoid(nn.Module):
        def __init__(self, alpha=4.0):
            super().__init__()
            self.alpha = alpha

        def forward(self, x):
            return _SigmoidSurrogate.apply(x, self.alpha)

    # Identical arithmetic to spikingjelly 0.0.0.0.15
    # LIFNode.jit_eval_multi_step_forward_hard_reset_decay_input.
    @torch.jit.script
    def _eval_ms_hard_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor,
                                        v_threshold: float, v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1.0 - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @torch.jit.script
    def _eval_ms_hard_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                   v_threshold: float, v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1.0 - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    class _BaseNode(nn.Module):
        """Minimal stand-in for spikingjelly.activation_based.neuron.BaseNode."""

        def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function=None,
                     detach_reset=False, step_mode="s", backend="torch", store_v_seq=False):
            super().__init__()
            self.v_threshold = v_threshold
            self.v_reset = v_reset
            self.surrogate_function = surrogate_function if surrogate_function is not None else Sigmoid()
            self.detach_reset = detach_reset
            self.step_mode = step_mode
            self.backend = backend  # accepted for API compatibility; cupy kernels are not ported
            self.store_v_seq = store_v_seq
            self.v = 0.0 if v_reset is None else v_reset
            self._v_init = self.v

        def reset(self):
            self.v = self._v_init
            if hasattr(self, "v_seq"):
                self.v_seq = None

        def v_float_to_tensor(self, x: torch.Tensor):
            if isinstance(self.v, float):
                self.v = torch.full_like(x.data, self.v)

        def neuronal_fire(self):
            return self.surrogate_function(self.v - self.v_threshold)

        def neuronal_reset(self, spike):
            spike_d = spike.detach() if self.detach_reset else spike
            if self.v_reset is None:
                self.v = self.v - self.v_threshold * spike_d
            else:
                self.v = spike_d * self.v_reset + (1.0 - spike_d) * self.v

        def single_step_forward(self, x):
            self.v_float_to_tensor(x)
            self.neuronal_charge(x)
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            return spike

        def _torch_multi_step_forward(self, x_seq):
            y_seq, v_seq = [], []
            for t in range(x_seq.shape[0]):
                y_seq.append(self.single_step_forward(x_seq[t]))
                if self.store_v_seq:
                    v_seq.append(self.v)
            if self.store_v_seq:
                self.v_seq = torch.stack(v_seq)
            return torch.stack(y_seq)

        def multi_step_forward(self, x_seq):
            return self._torch_multi_step_forward(x_seq)

        def forward(self, *args, **kwargs):
            if self.step_mode == "s":
                return self.single_step_forward(*args, **kwargs)
            elif self.step_mode == "m":
                return self.multi_step_forward(*args, **kwargs)
            raise ValueError(self.step_mode)

        def extra_repr(self):
            return (f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, "
                    f"detach_reset={self.detach_reset}, step_mode={self.step_mode}, "
                    f"backend={self.backend}")

    class LIFNode(_BaseNode):
        def __init__(self, tau: float = 2.0, decay_input: bool = True, v_threshold: float = 1.0,
                     v_reset: Optional[float] = 0.0, surrogate_function: Callable = None,
                     detach_reset: bool = False, step_mode="s", backend="torch",
                     store_v_seq: bool = False):
            assert isinstance(tau, float) and tau > 1.0
            super().__init__(v_threshold, v_reset, surrogate_function, detach_reset,
                             step_mode, backend, store_v_seq)
            self.tau = tau
            self.decay_input = decay_input

        def extra_repr(self):
            return super().extra_repr() + f", tau={self.tau}"

        def neuronal_charge(self, x: torch.Tensor):
            v_reset = 0.0 if self.v_reset is None else self.v_reset
            if self.decay_input:
                if v_reset == 0.0:
                    self.v = self.v + (x - self.v) / self.tau
                else:
                    self.v = self.v + (x - (self.v - v_reset)) / self.tau
            else:
                if v_reset == 0.0:
                    self.v = self.v * (1.0 - 1.0 / self.tau) + x
                else:
                    self.v = self.v - (self.v - v_reset) / self.tau + x

        def multi_step_forward(self, x_seq: torch.Tensor):
            if (not self.training) and self.v_reset is not None and self.decay_input:
                self.v_float_to_tensor(x_seq[0])
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = _eval_ms_hard_reset_decay_input_with_v_seq(
                        x_seq, self.v, float(self.v_threshold), float(self.v_reset), float(self.tau))
                else:
                    spike_seq, self.v = _eval_ms_hard_reset_decay_input(
                        x_seq, self.v, float(self.v_threshold), float(self.v_reset), float(self.tau))
                return spike_seq
            return self._torch_multi_step_forward(x_seq)

    class ParametricLIFNode(_BaseNode):
        def __init__(self, init_tau: float = 2.0, decay_input: bool = True, v_threshold: float = 1.0,
                     v_reset: Optional[float] = 0.0, surrogate_function: Callable = None,
                     detach_reset: bool = False, step_mode="s", backend="torch",
                     store_v_seq: bool = False):
            assert isinstance(init_tau, float) and init_tau > 1.0
            super().__init__(v_threshold, v_reset, surrogate_function, detach_reset,
                             step_mode, backend, store_v_seq)
            self.decay_input = decay_input
            init_w = -math.log(init_tau - 1.0)
            self.w = nn.Parameter(torch.as_tensor(init_w))

        def extra_repr(self):
            with torch.no_grad():
                tau = 1.0 / self.w.sigmoid()
            return super().extra_repr() + f", tau={tau}"

        def neuronal_charge(self, x: torch.Tensor):
            v_reset = 0.0 if self.v_reset is None else self.v_reset
            if self.decay_input:
                self.v = self.v + (x - (self.v - v_reset)) * self.w.sigmoid()
            else:
                self.v = self.v - (self.v - v_reset) * self.w.sigmoid() + x

    class _Functional:
        @staticmethod
        def reset_net(net: nn.Module):
            for m in net.modules():
                if hasattr(m, "reset"):
                    m.reset()

    functional = _Functional()


# ------------------------------------------------------------------ spikingjelly datasets
# spikingjelly >= 0.0.0.0.15 loads cached frames with .astype(np.float32); older releases
# return the npz array as stored (float64). float64 frames reach the model as DoubleTensor,
# which autocast does not cast, so the first conv fails ("Input type (DoubleTensor) and weight
# type (HalfTensor)"). The cast must happen in the loader, BEFORE the dataset's transform
# (e.g. dvs_utils.Resize interpolates), to reproduce the new behaviour exactly. Event counts
# are small integers, so float64 -> float32 is exact.
MIN_SJ_FRAMES_FLOAT32 = (0, 0, 0, 0, 15)
SJ_FRAMES_NEED_CAST = not _at_least(SJ_VERSION, MIN_SJ_FRAMES_FLOAT32)  # also True if unknown


class _Float32Loader:
    """Picklable wrapper (DataLoader workers) that casts a loaded frame array to float32."""

    def __init__(self, loader):
        self.loader = loader

    def __call__(self, path):
        import numpy as np
        x = self.loader(path)
        if isinstance(x, np.ndarray) and x.dtype != np.float32:
            x = x.astype(np.float32)
        return x


def _float32_frames(cls):
    if not SJ_FRAMES_NEED_CAST:
        return cls

    class Wrapped(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if hasattr(self, "loader") and not isinstance(self.loader, _Float32Loader):
                self.loader = _Float32Loader(self.loader)

    Wrapped.__name__ = Wrapped.__qualname__ = cls.__name__
    return Wrapped


from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS as _CIFAR10DVS  # noqa: E402
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture as _DVS128Gesture  # noqa: E402
from spikingjelly.datasets.n_caltech101 import NCaltech101 as _NCaltech101  # noqa: E402

CIFAR10DVS = _float32_frames(_CIFAR10DVS)
DVS128Gesture = _float32_frames(_DVS128Gesture)
NCaltech101 = _float32_frames(_NCaltech101)
