"""Unified, config-driven KL-damage calibration + per-expert pruning eval for ALL datasets
(cifar10/100 static + gesture/cifar10-dvs/ncaltech DVS). Builds the model and the train
(calibration) / test (eval) loaders exactly as firing_num.py, so it works for the DVS
pipelines the cifar-only sdt_alloc_calibrate.py / sdt_prune_eval.py cannot handle.

  # calibration (train split) -> npz consumed by alloc_solve.py
  python alloc_dvs.py --mode calib -c conf/gesture/4_192_truncprune.yml \
      --resume output/truncprune_gesture_4_192/model_best.pth.tar --use-ema \
      --num-cal 100000 --out alloc_calib_gesture.npz

  # eval (test split) at per-expert thresholds -> top-1 + active/4
  python alloc_dvs.py --mode eval -c conf/gesture/4_192_truncprune.yml \
      --resume output/truncprune_gesture_4_192/model_best.pth.tar --use-ema \
      --exit-threshold-per-expert "t0,...,t15"
"""
import argparse, os, yaml, numpy as np, torch
import torch.nn.functional as F
from timm.models import create_model
from timm.data import create_dataset, create_loader
import model as _model_reg                     # registers sdt
from module import ms_conv as moe_trunc
from sj_compat import functional, torch_load, compat_report
from sj_compat import CIFAR10DVS, DVS128Gesture
import dvs_utils

p = argparse.ArgumentParser()
p.add_argument("--mode", required=True, choices=["calib", "eval", "static", "load"])
p.add_argument("-c", "--config", required=True)
p.add_argument("--resume", required=True)
p.add_argument("--use-ema", action="store_true")
p.add_argument("--batch-size", type=int, default=None, help="override config; DVS defaults to 16")
p.add_argument("--num-cal", type=int, default=100000, help="max calibration images (calib mode)")
p.add_argument("--exit-threshold-per-expert", default=None, help="eval mode: L*E comma thresholds")
p.add_argument("--exit-threshold-per-layer", default=None, help="eval mode: L comma thresholds")
p.add_argument("--data-dir", default=None,
               help="override the config's data_dir (run_all.sh passes the resolved path)")
p.add_argument("--out", default="alloc_calib.npz")
args = p.parse_args()
print(compat_report())

# ---- read config ----
raw = yaml.safe_load(open(args.config))
cfg = {k.replace("-", "_"): v for k, v in raw.items()}
def g(k, d=None): return cfg.get(k, d)
DATASET = g("dataset"); DATA_DIR = args.data_dir or g("data_dir", "/scratch1/bkrhee/data")
NC = g("num_classes"); IMG = g("img_size"); INCH = g("in_channels", 3)
POOL = str(g("pooling_stat", "0011")); T = g("time_steps", 4); DIM = g("dim")
LAYER = g("layer"); MLP = float(g("mlp_ratio", 1)); NHEAD = g("num_heads", 8)
NE = g("num_experts", 4); TOPK = g("top_k", 1)
MRP = g("mlp_ratio_plain", MLP); PATCH = g("patch_size", None)
DVS_MODE = DATASET in ["cifar10-dvs-tet", "cifar10-dvs", "ncaltech101"]
IS_DVS = DATASET in dvs_utils.DVS_DATASET
BS = args.batch_size or (16 if IS_DVS else 64)
# mean/std may be absent/empty in the config (e.g. ImageNet -> model default_cfg = timm
# IMAGENET_DEFAULT); handle None and fall back to the right default per dataset.
_IN_MEAN, _IN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_m = g("mean") or (list(_IN_MEAN) if DATASET == "imagenet" else [0.5, 0.5, 0.5])
_s = g("std")  or (list(_IN_STD)  if DATASET == "imagenet" else [0.5, 0.5, 0.5])
MEAN = tuple(_m); STD = tuple(_s); CROP = float(g("crop_pct", 1.0))
INTERP = g("interpolation", "bilinear")   # match training resize (ImageNet uses bicubic)
AMP = bool(g("amp", False))               # match training-time validate autocast (fp16)
KEEP, FORCE = -1.0, 10.0
dev = "cuda"
print(f"[cfg] dataset={DATASET} dim={DIM} layer={LAYER} T={T} in_ch={INCH} nc={NC} "
      f"pool={POOL} dvs_mode={DVS_MODE} is_dvs={IS_DVS} bs={BS}")

# ---- model (identical to firing_num) ----
model = create_model("sdt", T=T, num_heads=NHEAD, num_classes=NC, pooling_stat=POOL,
    img_size_h=IMG, img_size_w=IMG, patch_size=PATCH, embed_dims=DIM, mlp_ratios=MLP,
    in_channels=INCH, qkv_bias=False, depths=LAYER, sr_ratios=1, spike_mode=g("spike_mode","lif"),
    dvs_mode=DVS_MODE, TET=False, mixing_mode=g("mixing_mode","none"), mix_ratio=g("mix_ratio",0.5),
    moe_type=g("moe_type","allrouted"), num_experts=NE, top_k=TOPK,
    use_output_lif=g("use_output_lif", False), alternating_moe=g("alternating_moe", False),
    mlp_ratio_plain=MRP).to(dev)
ck = torch_load(args.resume, map_location="cpu")
if args.use_ema and isinstance(ck, dict) and "state_dict_ema" in ck:
    sd = ck["state_dict_ema"]; print("  using EMA weights")
else:
    sd = ck.get("state_dict", ck)
miss, unexp = model.load_state_dict({k.replace("module.",""):v for k,v in sd.items()}, strict=False)
print(f"  loaded (missing {len(miss)}, unexpected {len(unexp)})")
model.eval()
moe = [blk.mlp for blk in model.block if hasattr(blk.mlp, "experts")]
L, E = len(moe), NE
for i, m in enumerate(moe):
    m.layer = i; m.early_exit = True; m.exit_metric = "input_entropy"
    m.prune_tmean = True; m.prune_only = True; m.exit_threshold = KEEP
    m.exit_threshold_per_expert = [KEEP] * E
print(f"  {L} MoE blocks x {E} experts")

# ---- data loader (train for calib, test for eval), matching firing_num ----
def build_loader(train):
    if DATASET == "cifar10-dvs":
        d = CIFAR10DVS(DATA_DIR, data_type="frame", frames_number=T, split_by="number",
                       transform=dvs_utils.Resize(64))
        tr, te = dvs_utils.split_to_train_test_set(0.9, d, 10); ds = tr if train else te
    elif DATASET == "ncaltech101":
        tr, te = dvs_utils.build_ncaltech(DATA_DIR, True); ds = tr if train else te
    elif DATASET == "gesture":
        ds = DVS128Gesture(DATA_DIR, train=train, data_type="frame", frames_number=T, split_by="number")
    else:  # static (torch/cifar10, torch/cifar100)
        ds = create_dataset(DATASET, root=DATA_DIR, split=("train" if train else "validation"),
                            is_training=False, download=True, batch_size=BS)
        return create_loader(ds, input_size=(INCH, IMG, IMG), batch_size=BS, is_training=False,
                             use_prefetcher=True, mean=MEAN, std=STD, num_workers=4, crop_pct=CROP,
                             interpolation=INTERP)
    return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True)

def fwd(x):
    out = model(x)
    if isinstance(out, tuple): out = out[0]
    if out.dim() == 3: out = out.mean(0)          # TET (T,B,cls)
    functional.reset_net(model)
    return out.float()

# =================== CALIBRATION ===================
if args.mode == "calib":
    loader = build_loader(train=True)
    ENT = [[[] for _ in range(E)] for _ in range(L)]
    KL  = [[[] for _ in range(E)] for _ in range(L)]
    FLP = [[[] for _ in range(E)] for _ in range(L)]
    n = 0
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            if n >= args.num_cal: break
            x = x.to(dev); n += x.size(0)
            moe_trunc.ENTROPY_CAPTURE = {}
            z = fwd(x); cap = moe_trunc.ENTROPY_CAPTURE; moe_trunc.ENTROPY_CAPTURE = None
            pf = F.softmax(z, 1); lpf = F.log_softmax(z, 1); pred = z.argmax(1)
            for l in range(L):
                for e in range(E):
                    ENT[l][e].append(cap[("te", l, e)][0].reshape(-1).cpu().numpy())
            for l in range(L):
                for e in range(E):
                    moe[l].exit_threshold_per_expert = [FORCE if j==e else KEEP for j in range(E)]
                    zp = fwd(x); moe[l].exit_threshold_per_expert = [KEEP]*E
                    kl = (pf * (lpf - F.log_softmax(zp,1))).sum(1)
                    KL[l][e].append(kl.cpu().numpy())
                    FLP[l][e].append((zp.argmax(1)!=pred).cpu().numpy())
            if bi % 8 == 0: print(f"  batch {bi}: {n} imgs", flush=True)
    out = {}
    for l in range(L):
        for e in range(E):
            out[f"ent_l{l}_e{e}"] = np.concatenate(ENT[l][e])
            out[f"kl_l{l}_e{e}"]  = np.concatenate(KL[l][e])
            out[f"flip_l{l}_e{e}"] = np.concatenate(FLP[l][e])
    np.savez(args.out, **out)
    print(f"saved {args.out}: {out['ent_l0_e0'].size} imgs x {L}x{E}")
    for l in range(L):
        print("  block%d: " % l + " ".join(f"e{e}:KL={out[f'kl_l{l}_e{e}'].mean():.4f}" for e in range(E)))

# =================== EXPERT LOAD ONLY (ranking export) ===================
# Same measurement as the first half of --mode static, but saves the (L,E) routing-load matrix
# and stops. Lets a caller rank experts globally (rather than per block) without paying for the
# per-block accuracy sweep that --mode static runs afterwards.
elif args.mode == "load":
    loader = build_loader(train=True)
    load = np.zeros((L, E)); n = 0
    with torch.no_grad():
        for x, _y in loader:
            if n >= args.num_cal: break
            x = x.to(dev); n += x.size(0)
            fwd(x)
            for l in range(L):
                load[l] += moe[l].last_expert_load.float().sum(0).cpu().numpy()
    load /= max(n, 1)
    np.savez(args.out, load=load)
    print(f"saved {args.out}: mean tokens/expert/sample over {n} train samples")
    for l in range(L):
        print(f"  block{l}: {np.round(load[l], 3).tolist()}")

# =================== STATIC EXPERT PRUNING (baseline) ===================
# Calibrate routing selection frequency on the TRAIN split, then permanently drop the k
# LEAST-selected experts in each block (k = 0..E) and evaluate. This is the "static" family
# the paper contrasts against: a fixed expert subset removed offline, reused for every input.
elif args.mode == "static":
    import contextlib
    _ac = torch.cuda.amp.autocast if AMP else contextlib.nullcontext
    loader = build_loader(train=True)
    load = np.zeros((L, E)); n = 0
    with torch.no_grad():
        for x, _y in loader:
            if n >= args.num_cal: break
            x = x.to(dev); n += x.size(0)
            fwd(x)
            for l in range(L):
                load[l] += moe[l].last_expert_load.float().sum(0).cpu().numpy()
    load /= max(n, 1)
    order = np.argsort(load, axis=1)                 # ascending: least-selected first
    print(f"\ncalibration on {n} train samples -- mean tokens/expert/sample:")
    for l in range(L):
        print(f"  block{l}: {np.round(load[l],2).tolist()}   least->most: {order[l].tolist()}")

    test = build_loader(train=False)
    print(f"\n{'pruned/blk':>11} {'top-1':>8} {'active/4':>9}   per-block active")
    for k in range(0, E + 1):
        for l in range(L):
            thr = [KEEP] * E
            for e in order[l][:k]: thr[e] = FORCE
            moe[l].exit_threshold_per_expert = thr
        correct = total = 0; active = np.zeros(L); nb = 0
        with torch.no_grad():
            for x, y in test:
                x = x.to(dev); y = y.to(dev)
                with _ac():
                    out = model(x)
                out = out[0] if isinstance(out, tuple) else out
                for bi, m in enumerate(moe):
                    te = getattr(getattr(m, "exit_module", None), "last_t_e", None)
                    active[bi] += te.gt(0).float().sum(1).mean().item() if te is not None else E
                functional.reset_net(model)
                if out.dim() == 3: out = out.mean(0)
                correct += (out.argmax(1) == y).sum().item(); total += y.numel(); nb += 1
        apb = active / max(nb, 1)
        tag = "unpruned" if k == 0 else (f"{k} least-selected" if k < E else f"{k} = ALL")
        print(f"{k:>11} {100.0*correct/total:>8.2f} {apb.mean():>9.3f}   "
              f"{np.round(apb,2).tolist()}  ({tag})")
    for l in range(L): moe[l].exit_threshold_per_expert = [KEEP] * E

# =================== EVAL ===================
else:
    if args.exit_threshold_per_expert:
        flat = [float(x) for x in args.exit_threshold_per_expert.split(",")]
        assert len(flat) == L*E, f"need {L*E} per-expert thr, got {len(flat)}"
        tpe = [flat[b*E:(b+1)*E] for b in range(L)]
        for i, m in enumerate(moe): m.exit_threshold_per_expert = tpe[i]
    elif args.exit_threshold_per_layer:
        thr = [float(x) for x in args.exit_threshold_per_layer.split(",")]
        for i, m in enumerate(moe): m.exit_threshold = thr[i]; m.exit_threshold_per_expert = None
    loader = build_loader(train=False)
    import contextlib
    _ac = torch.cuda.amp.autocast if AMP else contextlib.nullcontext
    correct = total = 0; active = np.zeros(L); nb = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev); y = y.to(dev)
            with _ac():
                out = model(x)
            out = out[0] if isinstance(out, tuple) else out
            for bi, m in enumerate(moe):
                te = getattr(getattr(m, "exit_module", None), "last_t_e", None)
                active[bi] += te.gt(0).float().sum(1).mean().item() if te is not None else E
            functional.reset_net(model)
            if out.dim() == 3: out = out.mean(0)
            correct += (out.argmax(1) == y).sum().item(); total += y.numel(); nb += 1
    apb = active / max(nb, 1)
    print(f"top-1: {100.0*correct/total:.2f}")
    print(f"average active experts (t_e>0): {apb.mean():.3f}/{E} (per-block {np.round(apb,2).tolist()})")
