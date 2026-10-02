# SDT + MoE — inference & calibration release

Spike-Driven Transformer with a Mixture-of-Experts feed-forward block, plus input-entropy
dynamic expert pruning. Everything needed to reproduce our reported accuracy on all six
datasets is in this directory; **only the dataset path needs to be configured.**

```
run_all.sh            ← (4) one script: set DATA_ROOT, runs inference on every dataset
manifest.json              per-dataset config, checkpoint, thresholds, expected accuracy
firing_num.py         ← (2) inference: top-1 accuracy + realized active-expert count
alloc_dvs.py          ← (2) calibration: per-expert input entropy + ablation KL
alloc_solve.py        ← (2) threshold allocation: knapsack over the calibration statistics
model/ module/ dvs_utils/  model code (registers `sdt`, the MoE block, DVS data loaders)
conf/<dataset>/*.yml       per-dataset config
checkpoints/<dataset>/     (3) fine-tuned SDT-MoE model at the deployed pruned operating point
checkpoints_unpruned/<dataset>/  the same model BEFORE pruning + recovery FT (the MoE baseline)
thresholds/<dataset>.txt   the per-expert thresholds, readable per block (also in manifest.json)
env/                  ← (1) the exact environment: conda environment.yml + pip requirements
```

## 1. Setup

Requires a GPU and the usual SNN stack: `torch`, `timm`, `spikingjelly`, `cupy`. Our runs used
Python 3.11, torch 2.7.1 (cu128), timm 1.0.26, spikingjelly 0.0.0.0.15.

**The exact environment ships in `env/`.** To build it with conda:

```bash
conda env create -p /path/to/envs/snn-gc -f env/environment.yml
conda activate /path/to/envs/snn-gc
```

It pins all 97 packages, taking only Python from conda and everything else from pip — torch is
the `+cu128` build from `download.pytorch.org` (the conda-forge and pytorch-channel builds are
different binaries, and fp16 spike timing is sensitive to that), and spikingjelly is pinned to
the exact commit we ran. For a plain pip/venv install, `env/requirements-cu128.txt` is the same
pin set:

```bash
pip install -r env/requirements-cu128.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

**Older stacks.** The code also runs on an older SNN environment (tested: Python 3.9,
torch 1.10, timm 0.5.4, spikingjelly 0.0.0.0.12, torchvision 0.11). `sj_compat.py` checks
the installed versions at import and supplies the few APIs those releases lack; the first
log line of each run (`compat: ...`) shows what was detected and what is shimmed. On the
pinned environment it resolves everything to the real libraries and changes nothing.
Expect accuracy within a few tenths of §3, as with any change of torch/cuDNN build.

`env/make_conda_env.sh` regenerates both from the source project's `uv.lock`. It is included
for provenance and is not needed to run anything here.

Place the datasets under a single root and point `DATA_ROOT` at it:

| dataset | expected path | layout |
|---|---|---|
| CIFAR-10 | `$DATA_ROOT` | torchvision (`cifar-10-batches-py/`) |
| CIFAR-100 | `$DATA_ROOT` | torchvision (`cifar-100-python/`) |
| ImageNet | `$DATA_ROOT/imagenet` | `train/`, `val/` in ImageFolder layout |
| CIFAR10-DVS | `$DATA_ROOT/cifar10-dvs` | spikingjelly root |
| DVS-Gesture | `$DATA_ROOT/DVSGesture` | spikingjelly root |
| N-Caltech101 | `$DATA_ROOT/NCALTECH101` | N-Caltech101 root |

Any single path can be overridden: `DATA_IMAGENET=/elsewhere/imagenet`.

The three event datasets must point at the **dataset directory itself, not its parent** —
spikingjelly looks for its extracted frames there and aborts with *"This dataset can not be
downloaded by SpikingJelly"* if given the parent. On first use it builds a frame cache inside
that directory, which takes a while; later runs reuse it.

## 2. Running inference on everything

```bash
cd /scratch2/bkrhee/sdt_moe_release
DATA_ROOT=/path/to/data bash run_all.sh
```

Subsets and options:

```bash
DATASETS="cifar100 gesture" bash run_all.sh    # only these
GPU=1 bash run_all.sh                          # pick a GPU
VENV=/path/to/.venv bash run_all.sh            # activate an env first
FORCE_EVAL=firing bash run_all.sh              # one harness for every dataset
```

**Two evaluation harnesses.** `manifest.json` carries an `eval` field per dataset: the three
event datasets use `alloc_dvs.py --mode eval`, which is the harness the reported numbers were
measured with, and the static ones use `firing_num.py`. The two differ only in how the test set
is batched — a plain `shuffle=False` DataLoader versus a shuffling `DistributedSampler` — but
under fp16 the batch an image lands in can flip a borderline spike, so top-1 can move by an
image (0.11 points on N-Caltech101's 914-image test set). `FORCE_EVAL` overrides the choice.

It prints a table of measured vs. expected top-1 and active-expert count, and writes one log
per dataset to `logs/`.

## 3. Expected results

Each model is fine-tuned at its deployed operating point. `active` is the mean number of
experts executed per block, out of 4 — averaged over the test set, since the decision is made
per input.

| dataset | model | top-1 | active /4 | pruned | unpruned MoE |
|---|---|---|---|---|---|
| CIFAR-10 | 4L-384 | **94.87** | 0.45 | 88.8% | 94.83 |
| CIFAR-100 | 4L-384 | **78.06** | 0.43 | 89.2% | 78.22 |
| ImageNet | 8L-768 | **73.34** | 1.96 | 51.0% | 73.54 |
| CIFAR10-DVS | 4L-192 | **76.50** | 0.54 | 86.5% | 77.10 |
| DVS-Gesture | 4L-192 | **98.61** | 1.01 | 74.5% | 97.22 |
| N-Caltech101 | 4L-192 | **76.70** | 0.67 | 83.2% | 76.81 |

Small deviations are expected: fp16 vs fp32 changes LIF spike timing slightly, which can move
top-1 by a few tenths. A large deviation usually means the data directory is not in the layout
above. The event datasets are evaluated with `--amp`, which `run_all.sh` applies automatically
from `manifest.json`.

## 4. How pruning works at inference

For each input, every expert computes the entropy of the tokens routed to it, and is skipped
when that entropy falls below its own threshold. The thresholds are per expert (16 values for
a 4-layer model, 32 for the 8-layer ImageNet model) and ship in `manifest.json`; `run_all.sh`
passes them as `--exit-threshold-per-expert`.

A value of `-1.0` means *never prune that expert*, which is why some threshold lists begin with
`-1.0000`. That is also why the flag is passed in `--flag=value` form — otherwise argparse
reads the leading minus as the start of a new option.

`thresholds/<dataset>.txt` holds the same values laid out one line per block, which is easier to
read than the flat string, and repeats the flat form at the bottom for copy-paste.

**The unpruned baseline.** `checkpoints_unpruned/<dataset>/` is the MoE model each deployed
checkpoint was fine-tuned from — trained with TET and the randomized per-expert timestep budget,
but with no pruning applied. It is the "unpruned MoE" column of the table in §3, and the right
reference for measuring what pruning costs. Score it by running the same command with no prune
flags:

```bash
python firing_num.py -c conf/gesture/4_192_finetune_pruned.yml \
  --model sdt --spike-mode lif --mlp-ratio 1 \
  -data-dir $DATA_ROOT/DVSGesture --resume checkpoints_unpruned/gesture/model_best.pth.tar \
  --no-resume-opt --no-prefetcher --amp --val-batch-size 16
```

Each one's provenance is recorded in `manifest.json` as `base_run` (the training run it came
from) and `base_file` (its path here).

## 5. Calibration and threshold allocation

The shipped thresholds were produced by two steps, both included so they can be re-run on a new
model or at a different pruning budget.

**Step 1 — calibrate.** Measures, for every (block, expert) pair, the input-entropy distribution
and the KL divergence caused by ablating that expert, over a sample of the training set:

```bash
python alloc_dvs.py --mode calib -c conf/cifar100/4_384_finetune_pruned.yml \
  --resume checkpoints/cifar100/model_best.pth.tar \
  --num-cal 4096 --out calib_cifar100.npz
```

**Step 2 — allocate.** Solves a knapsack over those statistics to pick each expert's threshold
under one global budget `B` (the target average prune rate), by Lagrangian relaxation with a
bisection on λ:

```bash
python alloc_solve.py --calib calib_cifar100.npz --arch sdt4_384 --cost uniform \
  --match-uniform 90 --out thresholds_cifar100.txt
```

`--arch` describes the model shape: `sdt4_384` (CIFAR-10/100), `sdt4_256` (event datasets),
`sdt8_768` (ImageNet). `--match-uniform` takes a comma-separated list of budgets as
percentages, so `25,50,75,90` emits one threshold row per budget.

The resulting `thr=` string from that file can be passed straight to `firing_num.py` as
`--exit-threshold-per-expert`, exactly as `run_all.sh` does with the shipped values.

Note that the **realized** prune rate is measured, not assumed: an expert marked for pruning
still runs on inputs whose entropy exceeds its threshold, so the realized rate is always
somewhat below the budget `B`. The `active` column in §3 is the measured value.
