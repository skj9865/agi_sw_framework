import argparse
import time
import yaml
import json
import os
import logging
import numpy as np
import random as rd
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from sj_compat import functional
from sj_compat import CIFAR10DVS, DVS128Gesture, NCaltech101
from sj_compat import LIFNode, ParametricLIFNode
import torch
import torch.nn as nn
import torchvision.utils
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    Mixup,
    FastCollateMixup,
    AugMixDataset,
)
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    load_checkpoint,
    model_parameters,
)
from sj_compat import clean_state_dict, torch_load, compat_report
from timm.utils import *
from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
    BinaryCrossEntropy,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
import model, dvs_utils

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

# DETERMINISTIC=1 trades throughput for run-to-run reproducibility: cudnn.benchmark picks
# conv algorithms by timing them, so the algorithm -- and its accumulation order -- can
# differ between runs and machines. In an SNN a last-bit change can flip a spike, which is
# why repeat evaluations can move top-1 by an image or two.
_DET = os.environ.get("DETERMINISTIC", "0") == "1"
torch.backends.cudnn.benchmark = not _DET
torch.backends.cudnn.deterministic = _DET
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="imagenet.yml",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument(
    "-data-dir",
    metavar="DIR",
    default="/scratch1/bkrhee/data",
    help="path to dataset",
)
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="torch/cifar10",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
parser.add_argument(
    "--train-split",
    metavar="NAME",
    default="train",
    help="dataset train split (default: train)",
)
parser.add_argument(
    "--val-split",
    metavar="NAME",
    default="validation",
    help="dataset validation split (default: validation)",
)
parser.add_argument(
    "--model",
    default="spikeformer",
    type=str,
    metavar="MODEL",
    help='Name of model to train (default: "countception")',
)
parser.add_argument(
    "--pooling-stat",
    default="1111",
    type=str,
    help="pooling layers in SPS moduls",
)
parser.add_argument(
    "--TET",
    default=False,
    type=bool,
    help="",
)
parser.add_argument(
    "--TET-means",
    default=1.0,
    type=float,
    help="",
)
parser.add_argument(
    "--TET-lamb",
    default=0.0,
    type=float,
    help="",
)
parser.add_argument(
    "--spike-mode",
    default="lif",
    type=str,
    help="",
)
parser.add_argument(
    "--mixing-mode",
    default="post_linear",
    choices=["none", "membrane", "post_linear"],
    help="gating mixing strategy: 'none' (plain LIF), 'membrane' (MultiStepMixingLIFNode), "
         "'post_linear' (token_mix_logit after gate_fc1) (default: post_linear)",
)
parser.add_argument(
    "--mix-ratio",
    default=0.1,
    type=float,
    help="initial mixing ratio (α) for membrane or post_linear mixing (default: 0.1)",
)
parser.add_argument(
    "--moe-type",
    default="moe",
    choices=["moe", "allrouted"],
    help="MoE variant: 'moe' (1 fixed + top-1 routed) or 'allrouted' (top-2 from all experts) (default: moe)",
)
parser.add_argument(
    "--num-experts",
    default=4,
    type=int,
    help="number of experts in MoE layers (default: 4)",
)
parser.add_argument(
    "--sample-routing",
    action="store_true",
    default=False,
    help="sample experts from softmax(router) during training (argmax at eval); default: deterministic top-k always",
)
parser.add_argument(
    "--use-ste",
    action="store_true",
    default=False,
    help="use Straight-Through Estimator for routing",
)
parser.add_argument(
    "--use-output-lif",
    action="store_true",
    default=False,
    help="add an additional LIF neuron at the output of each MoE expert",
)
parser.add_argument(
    "--alternating-moe",
    action="store_true",
    default=False,
    help="alternate MLP and MoE every other block (MLP-MoE-MLP-MoE...)",
)
parser.add_argument(
    "--mlp-ratio-plain",
    type=int,
    default=None,
    metavar="N",
    help="mlp_ratio for plain MLP blocks when using --alternating-moe; defaults to --mlp-ratio if not set",
)
parser.add_argument(
    "--early-exit",
    action="store_true",
    default=False,
    help="inference-only: per-token timestep early exit in the router (SEENN-style). "
         "Confident tokens (max gate softmax >= --exit-threshold) keep routed-expert output "
         "only for the first --exit-low-t timesteps",
)
parser.add_argument(
    "--exit-threshold",
    default=0.5,
    type=float,
    help="gate-softmax confidence threshold for early exit (default: 0.5)",
)
parser.add_argument(
    "--exit-low-t",
    default=1,
    type=int,
    help="number of timesteps confident tokens' routed experts run (default: 1)",
)
parser.add_argument(
    "--prune-threshold",
    default=None,
    type=float,
    help="if set, an expert group whose normalized routing entropy falls below this "
         "value is pruned (skipped, zero contribution). z-units when entropy norm is on "
         "(e.g. -0.5). default: None (no pruning)",
)
parser.add_argument(
    "--no-entropy-norm",
    action="store_true",
    default=False,
    help="disable per-layer z-score normalization of routing entropy; threshold then "
         "applies to raw [0,1] entropy instead",
)
parser.add_argument(
    "--exit-threshold-per-layer",
    default=None,
    type=str,
    help="override --exit-threshold with a comma-separated PER-MOE-BLOCK threshold "
         "(e.g. '0.5,0.98' = block0 conservative, block1 aggressive). Block 1 tolerates "
         "much more exit than block 0, so per-layer thresholds push the acc/t_e frontier. "
         "Length must equal the number of MoE blocks. default: None (use scalar threshold)",
)
parser.add_argument(
    "--exit-threshold-per-expert",
    default=None,
    type=str,
    help="comma-separated PER-(BLOCK,EXPERT) threshold, len=nblk*E in row-major order "
         "(b0e0,b0e1,...,b0e{E-1},b1e0,...). Used to deploy a sensitivity-allocated "
         "per-expert operating point. Overrides --exit-threshold-per-layer.",
)
parser.add_argument(
    "--exit-metric-per-layer",
    default=None,
    type=str,
    help="comma-separated PER-MOE-BLOCK exit metric (e.g. 'activity,output_entropy'). "
         "Motivated by Q2: spike-count separates block-0 experts, entropy separates block-1. "
         "default: None (use scalar --exit-metric)",
)
parser.add_argument(
    "--exit-mode-per-layer",
    default=None,
    type=str,
    help="comma-separated PER-MOE-BLOCK exit mode for input-side metrics "
         "(e.g. 'relmax,absolute'). default: None (use scalar --exit-mode)",
)
parser.add_argument(
    "--uniform-te",
    default=0,
    type=int,
    help="run EVERY routed expert at a fixed N timesteps (frozen after), no metric. "
         "0 = off. Cleanest deployment for a truncation-trained model.",
)
parser.add_argument(
    "--prune-experts",
    default="",
    type=str,
    help="comma-separated expert ids to PRUNE (skip entirely, zero output) in every MoE "
         "block, e.g. '3' or '1,3'. Tests pruning feasibility. default: none.",
)
parser.add_argument(
    "--expert-te",
    default="",
    type=str,
    help="per-expert fixed timestep budget (comma list, len=num_experts, same every block), "
         "e.g. '4,4,4,1' or '0,4,4,4' (0=prune). Sweeps the marginal value of one expert's steps.",
)
parser.add_argument(
    "--block-expert-te",
    default="",
    type=str,
    help="PER-BLOCK per-expert fixed budget; blocks separated by ';', experts by ',', "
         "e.g. '4,0,4,0;4,2,4,2' (block0 then block1; 0=prune). Gives each expert in each "
         "block its own timestep -> the per-expert adaptive schedule.",
)
parser.add_argument(
    "--prune-perimage-k",
    default=0,
    type=int,
    help="PER-IMAGE active-expert selection: for each image prune the K experts with the "
         "lowest routing load (keep the rest at full T). 0 = off. Which experts are dropped "
         "varies per image -> mean t_e = (E-K)/E * T.",
)
parser.add_argument(
    "--exit-prune-below",
    default=0,
    type=int,
    help="with the metric exit (output_entropy/convergence): if the metric would exit an "
         "expert at t_e < this value, PRUNE it (t_e=0) instead — pruning beats t_e=1. "
         "e.g. 2 maps metric exits of 1 -> 0. 0 = off.",
)
parser.add_argument(
    "--prune-only",
    action="store_true",
    default=False,
    help="pure prune-or-keep: the metric decides only keep-vs-prune; SURVIVORS run "
         "full T (no timestep reduction). Use with --exit-metric input_entropy "
         "--exit-prune-below 2 for input-entropy pruning with zero timestep change.",
)
parser.add_argument(
    "--prune-tmean",
    action="store_true",
    default=False,
    help="prune by the input entropy AVERAGED over all T timesteps (and tokens), not the "
         "t=1/cumulative value. Rule: prune iff T-averaged input entropy < --exit-threshold. "
         "Use with --exit-metric input_entropy --prune-only.",
)
parser.add_argument(
    "--prune-dvs-temporal",
    action="store_true",
    default=False,
    help="DVS variant of --prune-tmean: compute the input entropy PER TIMESTEP (softmax over "
         "channels at each t), mask out empty/no-event timesteps, then reduce over t — instead "
         "of the static mean-then-entropy form which destroys temporal structure and inflates "
         "entropy on sparse event data. Same rule: prune iff reduced entropy < --exit-threshold.",
)
parser.add_argument(
    "--prune-dvs-reduction",
    default="max",
    choices=["max", "mean"],
    help="temporal reduction for --prune-dvs-temporal: 'max' (prune only if every timestep is "
         "low-entropy; conservative) or 'mean' (event-activity-weighted mean over non-empty t).",
)
parser.add_argument(
    "--prune-spikecount",
    action="store_true",
    default=False,
    help="softmax-free variant of --prune-tmean: prune by the MEAN INPUT SPIKE RATE of each "
         "expert's dispatched input (sum of spikes / (tokens*channels*T)) instead of its entropy. "
         "Rule: prune iff mean firing rate > --exit-threshold (HIGH activity = already settled/"
         "low-entropy => skip the expert). Cheaper (no softmax/log). "
         "Use with --exit-metric input_entropy --prune-tmean --prune-only.",
)
parser.add_argument(
    "--prune-approx-entropy",
    action="store_true",
    default=False,
    help="prune by the entropy of an APPROXIMATED softmax of the expert input drive, using a cheap "
         "convex exp surrogate (base-2 shift or PWL, Softermax-style) instead of natural exp. "
         "Reproduces the true input_entropy's spread/discriminability with no exp/ln. Same prune "
         "direction (< --exit-threshold; flip with --prune-entropy-high). Use with --exit-metric "
         "input_entropy --prune-tmean --prune-only.",
)
parser.add_argument(
    "--approx-base", default="pwl",
    choices=["exp2", "pwl", "dyadic", "zonly", "meangap", "hardcount", "peakmean"],
    help="metric for --prune-approx-entropy. Full: 'exp2'/'pwl'/'dyadic' (approx softmax entropy). "
         "Ablation-ladder rungs (cheaper): 'zonly' (min-entropy log2Z), 'meangap' (linear-on-"
         "histogram), 'hardcount' (|{gap<delta}|, compare+popcount). All prune iff metric < "
         "--exit-threshold. Default pwl.",
)
parser.add_argument(
    "--approx-temp", type=float, default=None,
    help="temperature for --prune-approx-entropy (default ln2 for exp2/pwl/dyadic; 1.0 for the "
         "ladder rungs => /tau is a shift).",
)
parser.add_argument(
    "--approx-delta", type=float, default=None,
    help="for --approx-base hardcount: count channels with gap < delta (default 2). For meangap: "
         "the mean-gap normalizer kref (default 8).",
)
parser.add_argument(
    "--approx-kmax", type=int, default=None,
    help="for --approx-base dyadic/zonly: DISCARD channels with bit-gap > kmax (weight 0, NOT "
         "lumped) -- bounds the datapath to kmax+1 bins. Best ~5 (peaks the similarity to true "
         "entropy AND is cheapest). Default None = no cap (use all gaps).",
)
parser.add_argument(
    "--prune-spike-entropy",
    action="store_true",
    default=False,
    help="softmax-free SHAPE metric: prune by the entropy of each expert's per-CHANNEL spike "
         "histogram (counts -> distribution -> entropy), which keeps the channel-distribution "
         "shape that --prune-spikecount's scalar rate discards, without any exp/softmax. Same "
         "direction as input_entropy (prune iff < --exit-threshold; flip with --prune-entropy-high). "
         "Use with --exit-metric input_entropy --prune-tmean --prune-only.",
)
parser.add_argument(
    "--spike-entropy-base2",
    action="store_true",
    default=False,
    help="compute --prune-spike-entropy entropy LOG in the hardware form (log2 ~ MSB position, "
         "priority encoder); else exact natural-log (reference).",
)
parser.add_argument(
    "--spike-entropy-concentrate", default="linear", choices=["linear", "exp2"],
    help="distribution for --prune-spike-entropy: 'linear' p_c=n_c/N (plain histogram, saturates) "
         "or 'exp2' Softermax over counts p_c=2^{(n_c-max)/temp}/sum (concentrates; with temp=1 the "
         "2^{int} is an EXACT bit-shift -- spike-based Softermax, no PWL/exp).",
)
parser.add_argument(
    "--spike-entropy-temp", type=float, default=None,
    help="temperature for --spike-entropy-concentrate exp2 (default 1 => exact integer shifts).",
)
parser.add_argument(
    "--prune-spikecount-source",
    default="gate",
    choices=["gate", "expert_lif"],
    help="which spikes the --prune-spikecount rate is measured on. 'gate' (default): the gate_lif1 "
         "spikes dispatched per expert (free -- router runs anyway -- but one linear map removed "
         "from what the expert sees). 'expert_lif': the expert's OWN first LIF (fc1_lif of its "
         "dispatched input), the faithful 'what the expert processes' signal, still computed before "
         "fc1_conv so it costs no weight DMA.",
)
parser.add_argument(
    "--prune-zero-spikes",
    action="store_true",
    default=False,
    help="With --prune-spikecount: ALSO prune any expert whose dispatched spike rate is ~0 (a "
         "silent expert has nothing to process and only emits bias -> free to drop, skipping its "
         "weight fetch + compute). Applied as an OR on top of the rate>--exit-threshold rule, so "
         "set --exit-threshold high (e.g. 1.0) to prune ONLY silent experts. Useful on sparse "
         "routing (e.g. NCaltech/DVS) where most experts are silent per image.",
)
parser.add_argument(
    "--zero-spike-eps",
    type=float,
    default=1e-6,
    help="threshold below which a spike rate counts as 'silent' for --prune-zero-spikes (default 1e-6).",
)
parser.add_argument(
    "--prune-entropy-high",
    action="store_true",
    default=False,
    help="REVERSE the input-entropy prune direction: prune iff T-averaged input entropy > "
         "--exit-threshold (drop HIGH-entropy experts) instead of the default < (drop LOW-entropy). "
         "Use with --exit-metric input_entropy --prune-tmean --prune-only.",
)
parser.add_argument(
    "--exit-prune-metric",
    default=None,
    choices=["input_entropy", "input_convergence", "output_entropy", "output_convergence"],
    help="STACKED mode: use this metric (e.g. input_entropy) to decide keep-vs-PRUNE "
         "independently of --exit-metric (which then only sets the per-timestep budget on "
         "survivors). Lets you prune by input entropy AND shorten survivors' t_e by output "
         "entropy in one run. Prune fires when this metric exits at t_e < --exit-prune-below "
         "(default 2). default: None (single-metric mode).",
)
parser.add_argument(
    "--exit-prune-threshold",
    default=None,
    type=float,
    help="threshold for --exit-prune-metric (the prune decision). default: None -> reuse "
         "--exit-threshold.",
)
parser.add_argument(
    "--exit-prune-threshold-per-layer",
    default=None,
    type=str,
    help="comma-separated PER-MOE-BLOCK prune threshold for --exit-prune-metric, e.g. "
         "'0.80,0.97'. Overrides --exit-prune-threshold. Motivated by the entropy "
         "distribution: block 0 runs at much higher entropy (~0.74) than block 1 (~0.50), "
         "so one global threshold cannot separate both. default: None (scalar).",
)
parser.add_argument(
    "--exit-prune-metric-per-layer",
    default=None,
    type=str,
    help="comma-separated PER-MOE-BLOCK prune metric, e.g. 'input_convergence,input_entropy' "
         "(block 0 entropy is unseparated, convergence works better there). Overrides "
         "--exit-prune-metric. default: None (scalar).",
)
parser.add_argument(
    "--exit-metric",
    default="entropy",
    choices=["entropy", "activity", "both", "output_entropy", "output_convergence",
             "input_entropy", "input_convergence"],
    help="per-expert-group signal driving early exit/pruning: "
         "'entropy' (channel-softmax of router input), "
         "'activity' (mean spike count of gate_lif1 over T and C — slide-4 SA), "
         "'both' (average of the two z-scores), "
         "'output_entropy' (DT-SNN-style: entropy of each expert's accumulated OUTPUT "
         "after each timestep vs threshold; per-timestep exit). default: entropy",
)
parser.add_argument(
    "--exit-mode",
    default="absolute",
    choices=["absolute", "relmax"],
    help="early-exit threshold mode: 'absolute' (threshold vs z-scored/raw metric) or "
         "'relmax' (relative-to-max: each expert's spike activity / the most-active "
         "expert's in that layer; --exit-threshold is then alpha in [0,1], and the "
         "rule auto-adapts per layer). default: absolute",
)
parser.add_argument(
    "--layer",
    default=4,
    type=int,
    help="",
)
parser.add_argument(
    "--in-channels",
    default=3,
    type=int,
    help="",
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
parser.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Initialize model from this checkpoint (default: none)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
parser.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="prevent resume of optimizer state when resuming model",
)
parser.add_argument(
    "--num-classes",
    type=int,
    default=1000,
    metavar="N",
    help="number of label classes (Model default if None)",
)
parser.add_argument(
    "--time-steps",
    type=int,
    default=4,
    metavar="N",
    help="",
)
parser.add_argument(
    "--num-heads",
    type=int,
    default=8,
    metavar="N",
    help="",
)
parser.add_argument(
    "--patch-size", type=int, default=None, metavar="N", help="Image patch size"
)
parser.add_argument(
    "--mlp-ratio",
    type=int,
    default=4,
    metavar="N",
    help="expand ration of embedding dimension in MLP block",
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--img-size",
    type=int,
    default=None,
    metavar="N",
    help="Image patch size (default: None => model default)",
)
parser.add_argument(
    "--input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop percent (for validation only)",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "-vb",
    "--val-batch-size",
    type=int,
    default=16,
    metavar="N",
    help="input val batch size for training (default: 32)",
)

parser.add_argument(
    "--opt",
    default="sgd",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "sgd")',
)
parser.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: None, use opt default)",
)
parser.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay", type=float, default=0.0001, help="weight decay (default: 0.0001)"
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
parser.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='Gradient clipping mode. One of ("norm", "value", "agc")',
)

parser.add_argument(
    "--sched",
    default="step",
    type=str,
    metavar="SCHEDULER",
    help='LR scheduler (default: "step"',
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--lr-noise",
    type=float,
    nargs="+",
    default=None,
    metavar="pct, pct",
    help="learning rate noise on/off epoch percentages",
)
parser.add_argument(
    "--lr-noise-pct",
    type=float,
    default=0.67,
    metavar="PERCENT",
    help="learning rate noise limit percent (default: 0.67)",
)
parser.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="learning rate noise std-dev (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="learning rate cycle len multiplier (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-limit",
    type=int,
    default=1,
    metavar="N",
    help="learning rate cycle limit",
)
parser.add_argument(
    "--warmup-lr",
    type=float,
    default=0.0001,
    metavar="LR",
    help="warmup learning rate (default: 0.0001)",
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 2)",
)
parser.add_argument(
    "--epoch-repeats",
    type=float,
    default=0.0,
    metavar="N",
    help="epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).",
)
parser.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--decay-epochs",
    type=float,
    default=30,
    metavar="N",
    help="epoch interval to decay LR",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=3,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)
parser.add_argument(
    "--cooldown-epochs",
    type=int,
    default=10,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
)
parser.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10",
)
parser.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

parser.add_argument(
    "--no-aug",
    action="store_true",
    default=False,
    help="Disable all training augmentation, override other train aug args",
)
parser.add_argument(
    "--scale",
    type=float,
    nargs="+",
    default=[0.08, 1.0],
    metavar="PCT",
    help="Random resize scale (default: 0.08 1.0)",
)
parser.add_argument(
    "--ratio",
    type=float,
    nargs="+",
    default=[3.0 / 4.0, 4.0 / 3.0],
    metavar="RATIO",
    help="Random resize aspect ratio (default: 0.75 1.33)",
)
parser.add_argument(
    "--hflip", type=float, default=0.5, help="Horizontal flip training aug probability"
)
parser.add_argument(
    "--vflip", type=float, default=0.0, help="Vertical flip training aug probability"
)
parser.add_argument(
    "--color-jitter",
    type=float,
    default=0.4,
    metavar="PCT",
    help="Color jitter factor (default: 0.4)",
)
parser.add_argument(
    "--aa",
    type=str,
    default=None,
    metavar="NAME",
    help='Use AutoAugment policy. "v0" or "original". (default: None)',
),
parser.add_argument(
    "--aug-splits",
    type=int,
    default=0,
    help="Number of augmentation splits (default: 0, valid: 0 or >=2)",
)
parser.add_argument(
    "--jsd",
    action="store_true",
    default=False,
    help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.",
)
parser.add_argument(
    "--bce-loss",
    action="store_true",
    default=False,
    help="Enable BCE loss w/ Mixup/CutMix use.",
)
parser.add_argument(
    "--bce-target-thresh",
    type=float,
    default=None,
    help="Threshold for binarizing softened BCE targets (default: None, disabled)",
)
parser.add_argument(
    "--reprob",
    type=float,
    default=0.0,
    metavar="PCT",
    help="Random erase prob (default: 0.)",
)
parser.add_argument(
    "--remode", type=str, default="const", help='Random erase mode (default: "const")'
)
parser.add_argument(
    "--recount", type=int, default=1, help="Random erase count (default: 1)"
)
parser.add_argument(
    "--resplit",
    action="store_true",
    default=False,
    help="Do not random erase first (clean) augmentation split",
)
parser.add_argument(
    "--mixup",
    type=float,
    default=0.0,
    help="mixup alpha, mixup enabled if > 0. (default: 0.)",
)
parser.add_argument(
    "--cutmix",
    type=float,
    default=0.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 0.)",
)
parser.add_argument(
    "--cutmix-minmax",
    type=float,
    nargs="+",
    default=None,
    help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
)
parser.add_argument(
    "--mixup-prob",
    type=float,
    default=1.0,
    help="Probability of performing mixup or cutmix when either/both is enabled",
)
parser.add_argument(
    "--mixup-switch-prob",
    type=float,
    default=0.5,
    help="Probability of switching to cutmix when both mixup and cutmix enabled",
)
parser.add_argument(
    "--mixup-mode",
    type=str,
    default="batch",
    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
)
parser.add_argument(
    "--mixup-off-epoch",
    default=0,
    type=int,
    metavar="N",
    help="Turn off mixup after this epoch, disabled if 0 (default: 0)",
)
parser.add_argument(
    "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
)
parser.add_argument(
    "--train-interpolation",
    type=str,
    default="random",
    help='Training interpolation (random, bilinear, bicubic default: "random")',
)
parser.add_argument(
    "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)"
)
parser.add_argument(
    "--drop-connect",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop connect rate, DEPRECATED, use drop-path (default: None)",
)
parser.add_argument(
    "--drop-path",
    type=float,
    default=0.2,
    metavar="PCT",
    help="Drop path rate (default: None)",
)
parser.add_argument(
    "--drop-block",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop block rate (default: None)",
)

parser.add_argument(
    "--bn-tf",
    action="store_true",
    default=False,
    help="Use Tensorflow BatchNorm defaults for models that support it (default: False)",
)
parser.add_argument(
    "--bn-momentum",
    type=float,
    default=None,
    help="BatchNorm momentum override (if not None)",
)
parser.add_argument(
    "--bn-eps",
    type=float,
    default=None,
    help="BatchNorm epsilon override (if not None)",
)
parser.add_argument(
    "--sync-bn",
    action="store_true",
    help="Enable NVIDIA Apex or Torch synchronized BatchNorm.",
)
parser.add_argument(
    "--dist-bn",
    type=str,
    default="",
    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")',
)
parser.add_argument(
    "--split-bn",
    action="store_true",
    help="Enable separate BN layers per augmentation split.",
)

parser.add_argument(
    "--model-ema",
    action="store_true",
    default=False,
    help="Enable tracking moving average of model weights",
)
parser.add_argument(
    "--model-ema-force-cpu",
    action="store_true",
    default=False,
    help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.",
)
parser.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.9998,
    help="decay factor for model weights moving average (default: 0.9998)",
)

parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint",
)
parser.add_argument(
    "--checkpoint-hist",
    type=int,
    default=10,
    metavar="N",
    help="number of checkpoints to keep (default: 10)",
)
parser.add_argument(
    "-j",
    "--workers",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 1)",
)
parser.add_argument(
    "--save-images",
    action="store_true",
    default=False,
    help="save images of input bathes every log interval for debugging",
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use NVIDIA Apex AMP or Native AMP for mixed precision training",
)
parser.add_argument(
    "--apex-amp",
    action="store_true",
    default=False,
    help="Use NVIDIA Apex AMP mixed precision",
)
parser.add_argument(
    "--native-amp",
    action="store_true",
    default=False,
    help="Use Native Torch AMP mixed precision",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--dvs-aug",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--dvs-trival-aug",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--save-qkv",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)

parser.add_argument(
    "--print-moe",
    action="store_true",
    default=False,
    help="print MoE expert routing summary during inference",
)
parser.add_argument(
    "--print-moe-batches",
    type=int,
    default=1,
    metavar="N",
    help="number of batches to print MoE routing (default: 1)",
)
parser.add_argument(
    "--print-moe-detail",
    action="store_true",
    default=False,
    help="print per-token MoE routing with full softmax distribution",
)
parser.add_argument(
    "--print-moe-max-tokens",
    type=int,
    default=None,
    metavar="N",
    help="max tokens to print per module (default: all tokens)",
)
parser.add_argument(
    "--max-batches",
    type=int,
    default=None,
    metavar="N",
    help="limit number of validation batches (default: all)",
)
parser.add_argument(
    "--first-sample-only",
    action="store_true",
    default=False,
    help="use only the first sample in each batch (index 0)",
)

parser.add_argument(
    "--output",
    default="",
    type=str,
    metavar="PATH",
    help="path to output folder (default: none, current dir)",
)
parser.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)
parser.add_argument(
    "--eval-metric",
    default="top1",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric (default: "top1")',
)
parser.add_argument(
    "--tta",
    type=int,
    default=0,
    metavar="N",
    help="Test/inference time augmentation (oversampling) factor. 0=None (default: 0)",
)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument(
    "--use-multi-epochs-loader",
    action="store_true",
    default=False,
    help="use the multi-epochs-loader to save time at the beginning of every epoch",
)
parser.add_argument(
    "--large-valid",
    action="store_true",
    default=False,
    help="use the multi-epochs-loader to save time at the beginning of every epoch",
)
parser.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="convert model torchscript for inference",
)
parser.add_argument(
    "--log-wandb",
    action="store_true",
    default=False,
    help="log training and validation metrics to wandb",
)

_logger = logging.getLogger("valid")
stream_handler = logging.StreamHandler()
format_str = "%(asctime)s %(levelname)s: %(message)s"
stream_handler.setFormatter(logging.Formatter(format_str))
_logger.addHandler(stream_handler)
_logger.propagate = False


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            cfg = {k.replace("-", "_"): v for k, v in cfg.items()}
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    _logger.info(compat_report())
    args, args_text = _parse_args()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:1"
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        _logger.info("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    use_amp = None
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = "apex"
    elif args.native_amp and has_native_amp:
        use_amp = "native"
    elif args.apex_amp or args.native_amp:
        _logger.warning(
            "Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDA apex or upgrade to PyTorch 1.6"
        )

    torch.backends.cudnn.benchmark = not _DET      # see DETERMINISTIC at the top
    torch.backends.cudnn.deterministic = _DET
    if _DET:
        torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.initial_seed()  # dataloader multi processing
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random_seed(args.seed, args.rank)
    rd.seed(args.seed)

    args.dvs_mode = args.dataset in ["cifar10-dvs-tet", "cifar10-dvs", "ncaltech101"]

    model = create_model(
        args.model,
        T=args.time_steps,
        pretrained=args.pretrained,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        pooling_stat=args.pooling_stat,
        img_size_h=args.img_size,
        img_size_w=args.img_size,
        patch_size=args.patch_size,
        embed_dims=args.dim,
        mlp_ratios=args.mlp_ratio,
        in_channels=args.in_channels,
        qkv_bias=False,
        depths=args.layer,
        sr_ratios=1,
        spike_mode=args.spike_mode,
        dvs_mode=args.dvs_mode,
        TET=args.TET,
        mixing_mode=args.mixing_mode,
        mix_ratio=args.mix_ratio,
        moe_type=args.moe_type,
        num_experts=args.num_experts,
        top_k=getattr(args, "top_k", None),  # MUST match training: allrouted default is top-2,
        sample_routing=args.sample_routing,
        use_ste=args.use_ste,
        use_output_lif=args.use_output_lif,
        alternating_moe=args.alternating_moe,
        mlp_ratio_plain=args.mlp_ratio_plain if args.mlp_ratio_plain is not None else args.mlp_ratio,
        early_exit=args.early_exit,
        exit_threshold=args.exit_threshold,
        exit_low_T=args.exit_low_t,
        prune_threshold=args.prune_threshold,
        entropy_norm=not args.no_entropy_norm,
        exit_metric=args.exit_metric,
        exit_mode=args.exit_mode,
    )
    moe_blocks = [blk for blk in model.block if hasattr(blk.mlp, "exit_threshold")]
    if getattr(args, "uniform_te", 0):
        N = int(args.uniform_te)
        for blk in moe_blocks:
            blk.mlp.expert_timesteps = [N] * blk.mlp.num_experts
        if args.local_rank == 0:
            _logger.info(f"  uniform fixed per-expert budget t_e={N} (no metric)")
    if getattr(args, "expert_te", ""):
        te = [int(x) for x in args.expert_te.split(",")]
        for blk in moe_blocks:
            assert len(te) == blk.mlp.num_experts, "expert-te length must equal num_experts"
            blk.mlp.expert_timesteps = list(te)
        if args.local_rank == 0:
            _logger.info(f"  per-expert fixed budget t_e={te} (0=prune)")
    if getattr(args, "block_expert_te", ""):
        per_block = [[int(x) for x in b.split(",")] for b in args.block_expert_te.split(";")]
        assert len(per_block) == len(moe_blocks), "block-expert-te needs one list per MoE block"
        flat = []
        for blk, te in zip(moe_blocks, per_block):
            assert len(te) == blk.mlp.num_experts, "each block list must have num_experts values"
            blk.mlp.expert_timesteps = list(te); flat += te
        if args.local_rank == 0:
            _logger.info(f"  per-block per-expert t_e={per_block} (0=prune)  mean t_e={sum(flat)/len(flat):.3f}")
    if getattr(args, "exit_prune_below", 0):
        for blk in moe_blocks:
            blk.mlp.exit_prune_below = int(args.exit_prune_below)
        if args.local_rank == 0:
            _logger.info(f"  metric exit: prune when t_e < {args.exit_prune_below}")
    if getattr(args, "prune_only", False):
        for blk in moe_blocks:
            blk.mlp.prune_only = True
        if args.local_rank == 0:
            _logger.info("  prune-only: survivors run full T (no timestep reduction)")
    if getattr(args, "prune_tmean", False):
        for blk in moe_blocks:
            blk.mlp.prune_tmean = True
        if args.local_rank == 0:
            _logger.info("  prune-tmean: prune iff T-averaged input entropy < exit_threshold")
    if getattr(args, "prune_dvs_temporal", False):
        red = getattr(args, "prune_dvs_reduction", "max")
        for blk in moe_blocks:
            blk.mlp.prune_dvs_temporal = True
            blk.mlp.prune_dvs_reduction = red
        if args.local_rank == 0:
            _logger.info(f"  prune-dvs-temporal: per-timestep input entropy (reduction={red}), "
                         f"empty-frame masked")
    if getattr(args, "prune_spikecount", False):
        src = getattr(args, "prune_spikecount_source", "gate")
        for blk in moe_blocks:
            blk.mlp.prune_spikecount = True
            blk.mlp.prune_spikecount_source = src
        if args.local_rank == 0:
            _logger.info(f"  prune-spikecount: prune iff mean input spike rate > exit_threshold "
                         f"(HIGH=prune), source={src}")
    if getattr(args, "prune_approx_entropy", False):
        ab = getattr(args, "approx_base", "pwl")
        at = getattr(args, "approx_temp", None)
        ad = getattr(args, "approx_delta", None)
        ak = getattr(args, "approx_kmax", None)
        for blk in moe_blocks:
            blk.mlp.prune_approx_entropy = True
            blk.mlp.approx_base = ab
            blk.mlp.approx_temp = at
            blk.mlp.approx_delta = ad
            blk.mlp.approx_kmax = ak
        if args.local_rank == 0:
            _logger.info(f"  prune-approx-entropy: prune iff approx-softmax({ab}) input entropy < "
                         f"exit_threshold (LOW=prune), temp={at if at is not None else 'ln2'}")
    if getattr(args, "prune_spike_entropy", False):
        src = getattr(args, "prune_spikecount_source", "gate")
        b2 = bool(getattr(args, "spike_entropy_base2", False))
        conc = getattr(args, "spike_entropy_concentrate", "linear")
        stmp = getattr(args, "spike_entropy_temp", None)
        for blk in moe_blocks:
            blk.mlp.prune_spike_entropy = True
            blk.mlp.prune_spikecount_source = src
            blk.mlp.spike_entropy_base2 = b2
            blk.mlp.spike_entropy_concentrate = conc
            blk.mlp.spike_entropy_temp = stmp
        if args.local_rank == 0:
            _logger.info(f"  prune-spike-entropy: prune iff per-channel spike entropy < exit_threshold "
                         f"(LOW=prune), source={src}, concentrate={conc}, temp={stmp}, base2={b2}")
    if getattr(args, "prune_zero_spikes", False):
        for blk in moe_blocks:
            blk.mlp.prune_zero_spikes = True
            blk.mlp.zero_spike_eps = float(getattr(args, "zero_spike_eps", 1e-6))
        if args.local_rank == 0:
            _logger.info(f"  prune-zero-spikes: ALSO prune iff spike rate <= {getattr(args, 'zero_spike_eps', 1e-6)} "
                         "(silent expert, free); composes with the spike-count threshold")
    if getattr(args, "prune_entropy_high", False):
        for blk in moe_blocks:
            blk.mlp.prune_entropy_high = True
        if args.local_rank == 0:
            _logger.info("  prune-entropy-high: prune iff T-mean input entropy > exit_threshold (HIGH=prune)")
    pl_pmet = [x.strip() for x in args.exit_prune_metric_per_layer.split(",")] \
        if getattr(args, "exit_prune_metric_per_layer", None) else None
    pl_pthr = [float(x) for x in args.exit_prune_threshold_per_layer.split(",")] \
        if getattr(args, "exit_prune_threshold_per_layer", None) else None
    for pl, what in ((pl_pmet, "exit-prune-metric-per-layer"), (pl_pthr, "exit-prune-threshold-per-layer")):
        if pl is not None:
            assert len(pl) == len(moe_blocks), \
                f"{what} has {len(pl)} values but model has {len(moe_blocks)} MoE blocks"
    if pl_pmet or pl_pthr or getattr(args, "exit_prune_metric", None):
        scalar_thr = args.exit_prune_threshold if args.exit_prune_threshold is not None else args.exit_threshold
        for i, blk in enumerate(moe_blocks):
            metric_i = pl_pmet[i] if pl_pmet else args.exit_prune_metric
            thr_i = pl_pthr[i] if pl_pthr else scalar_thr
            if metric_i is None:   # per-layer threshold given without a metric -> need a metric
                continue
            blk.mlp.exit_prune_metric = metric_i
            blk.mlp.exit_prune_threshold = float(thr_i)
        if args.local_rank == 0:
            _logger.info(f"  STACKED: prune metric={pl_pmet or args.exit_prune_metric} "
                         f"th={pl_pthr or scalar_thr}; timestep budget by {args.exit_metric} "
                         f"(th={args.exit_threshold})")
    if getattr(args, "prune_perimage_k", 0):
        for blk in moe_blocks:
            blk.mlp.prune_perimage_k = int(args.prune_perimage_k)
        if args.local_rank == 0:
            _logger.info(f"  per-image active-expert selection: prune K={args.prune_perimage_k} lowest-load/image")
    if getattr(args, "prune_experts", ""):
        pruned = [int(x) for x in args.prune_experts.split(",") if x.strip() != ""]
        for blk in moe_blocks:
            E = blk.mlp.num_experts
            blk.mlp.only_expert_ids = [e for e in range(E) if e not in pruned]
        if args.local_rank == 0:
            _logger.info(f"  pruning experts {pruned} (kept {moe_blocks[0].mlp.only_expert_ids})")
    def _per_layer(arg, cast):
        if arg is None:
            return None
        vals = [cast(x.strip()) for x in arg.split(",")]
        assert len(vals) == len(moe_blocks), (
            f"per-layer arg has {len(vals)} values but model has {len(moe_blocks)} MoE blocks")
        return vals
    pl_thr = _per_layer(args.exit_threshold_per_layer, float)
    pl_met = _per_layer(args.exit_metric_per_layer, str)
    pl_mode = _per_layer(args.exit_mode_per_layer, str)
    # per-(block,expert) thresholds override per-layer (deploy a sensitivity-allocated point)
    pe_thr = None
    if args.exit_threshold_per_expert:
        flat = [float(x.strip()) for x in args.exit_threshold_per_expert.split(",")]
        nblk = len(moe_blocks)
        assert len(flat) % nblk == 0, (
            f"--exit-threshold-per-expert has {len(flat)} values, not divisible by {nblk} blocks")
        E_pe = len(flat) // nblk
        pe_thr = [flat[b * E_pe:(b + 1) * E_pe] for b in range(nblk)]
        for i, blk in enumerate(moe_blocks):
            blk.mlp.exit_threshold_per_expert = pe_thr[i]
    if pl_thr or pl_met or pl_mode:
        for i, blk in enumerate(moe_blocks):
            mlp = blk.mlp
            if pl_thr is not None and pe_thr is None:
                mlp.exit_threshold = pl_thr[i]
                mlp.exit_module.entropy_threshold = pl_thr[i]
            if pl_met is not None:
                mlp.exit_metric = pl_met[i]
                if pl_met[i] not in ("output_entropy", "output_convergence",
                                     "input_entropy", "input_convergence"):
                    mlp.exit_module.metric = pl_met[i]
            if pl_mode is not None:
                mlp.exit_module.exit_mode = pl_mode[i]
        if args.local_rank == 0:
            _logger.info(f"  per-layer exit: metric={pl_met} mode={pl_mode} threshold={pl_thr}")

    if args.local_rank == 0:
        _logger.info("Creating model")
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _logger.info(f"number of params: {n_parameters}")
        _logger.info(
            f"  early_exit={args.early_exit}, exit_metric={args.exit_metric}, "
            f"exit_mode={args.exit_mode}, "
            f"exit_threshold={args.exit_threshold}, exit_low_t={args.exit_low_t}, "
            f"prune_threshold={args.prune_threshold}, entropy_norm={not args.no_entropy_norm}"
        )

    if args.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = (
            model.num_classes
        )  # FIXME handle model default vs config num_classes more elegantly

    if args.local_rank == 0:
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    data_config = resolve_data_config(
        vars(args), model=model, verbose=args.local_rank == 0
    )

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != "native":
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    if args.torchscript:
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        assert not args.sync_bn, "Cannot use SyncBatchNorm with torchscripted model"
        model = torch.jit.script(model)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if args.local_rank == 0:
            _logger.info("AMP not enabled. Training in float32.")

    if args.resume:
        checkpoint = torch_load(args.resume, map_location='cpu')
        # Prefer EMA weights automatically. A training/FT checkpoint stores EMA under
        # 'state_dict_ema'; when present (and not already folded into state_dict via
        # ema_extracted) we evaluate the EMA model, which is the reported/deployed one.
        if 'state_dict_ema' in checkpoint and not checkpoint.get('ema_extracted', False):
            state_dict = clean_state_dict(checkpoint['state_dict_ema'])
            if args.local_rank == 0:
                _logger.info("Loaded EMA weights (state_dict_ema) for inference")
        else:
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if args.local_rank == 0 and missing:
            _logger.warning(f"Missing keys (will use random init): {missing}")
        if args.local_rank == 0 and unexpected:
            _logger.warning(f"Unexpected keys (ignored): {unexpected}")

    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    if args.distributed:
        if has_apex and use_amp != "native":
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True, find_unused_parameters=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(
                model, device_ids=[args.local_rank], find_unused_parameters=True
            )  # can use device str in Torch >= 1.1

    dataset_eval = None, None
    if args.dataset == "cifar10-dvs-tet":
        dataset_eval = dvs_utils.DVSCifar10(
            root=os.path.join(args.data_dir, "test"),
            train=False,
        )
    elif args.dataset == "cifar10-dvs":
        dataset = CIFAR10DVS(
            args.data_dir,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
            transform=dvs_utils.Resize(64),   # MUST match train.py (model trained on 64x64)
        )
        _, dataset_eval = dvs_utils.split_to_train_test_set(0.9, dataset, 10)
    elif args.dataset == "ncaltech101":
        _, dataset_eval = dvs_utils.build_ncaltech(args.data_dir, True)
    elif args.dataset == "gesture":
        dataset_eval = DVS128Gesture(
            args.data_dir,
            train=False,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
        )
    else:
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            batch_size=args.batch_size,
        )

    loader_eval = None
    if args.dataset in dvs_utils.DVS_DATASET:
        loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    elif args.dataset == "imagenet" and args.large_valid:
        dataset_eval.transform = transforms.Compose(
            [
                transforms.Resize(320),
                transforms.CenterCrop(288),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
            ]
        )
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_eval)
        loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=args.val_batch_size,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config["input_size"],
            batch_size=args.val_batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config["crop_pct"],
            pin_memory=args.pin_mem,
        )

    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = "-".join(
            [
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                "data-" + args.dataset.split("/")[-1],
                f"t-{args.time_steps}",
                f"spike-{args.spike_mode}",
            ]
        )
    output_dir = get_outdir(args.output if args.output else "./output/valid", exp_name)
    if args.rank == 0:
        file_handler = logging.FileHandler(
            os.path.join(output_dir, f"{args.model}.log"), "w"
        )
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(logging.INFO)
        _logger.addHandler(file_handler)

    try:
        if args.distributed and args.dist_bn in ("broadcast", "reduce"):
            if args.local_rank == 0:
                _logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == "reduce")
        

        eval_metrics = validate(
            model,
            loader_eval,
            validate_loss_fn,
            args,
            output_dir=output_dir,
            amp_autocast=amp_autocast,
        )
        if args.local_rank == 0:
            non_zero_str = json.dumps(eval_metrics["non_zero"], indent=4)
            firing_rate_str = json.dumps(eval_metrics["firing_rate"], indent=4)
            _logger.info("top-1: %s", eval_metrics["top1"])
            _logger.info("non_zero: ")
            _logger.info(non_zero_str)
            _logger.info("firing_rate: ")
            _logger.info(firing_rate_str)
        if model_ema is not None and not args.model_ema_force_cpu:
            if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                distribute_bn(model_ema, args.world_size, args.dist_bn == "reduce")
    except KeyboardInterrupt:
        pass


def validate(
    model, loader, loss_fn, args, output_dir=None, amp_autocast=suppress, log_suffix=""
):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    def _time_slice(v_, t, T):
        if not torch.is_tensor(v_):
            return None

        if v_.dim() == 0:
            return None

        if v_.shape[0] != T:
            return None

        return v_[t]


    def calc_non_zero_rate(s_dict, nz_dict, denom, t, T):
        for k, v_ in s_dict.items():
            v = _time_slice(v_, t, T)
            if v is None:
                continue


            numel = v.numel()
            if numel == 0:
                continue
            nz = torch.count_nonzero(v).item()
            nz_dict[k] = nz_dict.get(k, 0.0) + (nz / numel) / denom
        return nz_dict


    def calc_firing_rate(s_dict, fr_dict, denom, t, T):
        for k, v_ in s_dict.items():
            v = _time_slice(v_, t, T)
            if v is None:
                continue

            fr_dict[k] = fr_dict.get(k, 0.0) + v.mean().item() / denom
        return fr_dict

    def log_moe_routing(batch_idx):
        from module.ms_conv import Top2Gating

        
        if not args.print_moe or args.local_rank != 0:
            return
        if args.print_moe_batches is not None and batch_idx >= args.print_moe_batches:
            return
        for name, module in model.named_modules():
            if not isinstance(module, Top2Gating):
                continue
            if module.last_masks is None:
                continue
            for k, mask in enumerate(module.last_masks):
                counts = (
                    mask.sum(dim=(0, 1)).to(torch.int64).detach().cpu().tolist()
                )
                print(f"[batch {batch_idx}] {name} top{k + 1} counts: {counts}")
            if args.print_moe_detail and module.last_raw_gates is not None:
                probs = module.last_raw_gates.detach().cpu()  # (B, N, E)
                indices = module.last_indices
                if indices is not None:
                    indices = [idx.detach().cpu() for idx in indices]
                B, N, E = probs.shape
                max_tokens = args.print_moe_max_tokens
                token_idx = 0
                for b in range(B):
                    for n in range(N):
                        if max_tokens is not None and token_idx >= max_tokens:
                            return
                        token_idx += 1
                        topk = []
                        if indices is not None:
                            for k, idx in enumerate(indices):
                                topk.append(
                                    {
                                        "k": k + 1,
                                        "expert": int(idx[b, n].item()),
                                        "prob": float(probs[b, n, idx[b, n]].item()),
                                    }
                                )
                        print(
                            f"[batch {batch_idx}] {name} token b{b} n{n} topk={topk} probs={probs[b, n].tolist()}"
                        )


    model.eval()
    
    end = time.time()
    
    max_batches = args.max_batches
    if max_batches is None:
        last_idx = len(loader) - 1
    else:
        last_idx = min(len(loader), max_batches) - 1

    
    fr_dict = {f"t{i}": dict() for i in range(args.time_steps)}
    nz_dict = {f"t{i}": dict() for i in range(args.time_steps)}

    from module.ms_conv import EarlyExitMeter, EntropyProbe
    ee_meter = EarlyExitMeter()
    ent_probe = EntropyProbe()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            denom= len(loader)

            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
            target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            if args.first_sample_only:
                input = input[:1]
                target = target[:1]

            with amp_autocast():
                output, firing_dict = model(input, hook=dict())
                if args.save_qkv and args.local_rank == 0:
                    torch.save(
                        firing_dict, os.path.join(output_dir, f"qkv_{batch_idx}.pkl")
                    )

                log_moe_routing(batch_idx)
                ee_meter.update(model)
                ent_probe.update(model)

            for t in range(args.time_steps):
                fr_dict[f"t{t}"] = calc_firing_rate(firing_dict, fr_dict[f"t{t}"], denom, t, args.time_steps)
                nz_dict[f"t{t}"] = calc_non_zero_rate(firing_dict, nz_dict[f"t{t}"], denom, t, args.time_steps)

            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]

            loss = loss_fn(output, target)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (
                last_batch or batch_idx % args.log_interval == 0
            ):
                log_name = "Test" + log_suffix
                _logger.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m,
                    )
                )

    if args.local_rank == 0:
        ee_meter.report()
        ent_probe.report()

    metrics = OrderedDict(
        [
            ("loss", losses_m.avg),
            ("top1", top1_m.avg),
            ("top5", top5_m.avg),
            ("non_zero", nz_dict),
            ("firing_rate", fr_dict),
        ]
    )

    return metrics


if __name__ == "__main__":
    main()

