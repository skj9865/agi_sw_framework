#!/bin/bash
# ============================================================================================
# SDT + MoE — inference on every dataset, reporting top-1 accuracy and the realized number of
# active experts at the deployed operating point.
#
#   1. set DATA_ROOT below (or export it), 2. run:
#
#        bash run_all.sh
#        DATA_ROOT=/my/data bash run_all.sh
#        DATASETS="cifar100 gesture" bash run_all.sh        # a subset
#        DATA_IMAGENET=/elsewhere/imagenet bash run_all.sh  # one dataset in a different place
#
# Nothing else needs editing: the model shape, the per-expert thresholds and the extra flags
# each dataset needs are read from manifest.json, and the checkpoints ship in checkpoints/.
#
# EXPECTED DATA LAYOUT under $DATA_ROOT (override any of them individually with DATA_<NAME>):
#
#   cifar10      $DATA_ROOT                      torchvision layout (cifar-10-batches-py/)
#   cifar100     $DATA_ROOT                      torchvision layout (cifar-100-python/)
#   imagenet     $DATA_ROOT/imagenet             train/ and val/ in ImageFolder layout
#   cifar10dvs   $DATA_ROOT/cifar10-dvs          spikingjelly CIFAR10-DVS root
#   gesture      $DATA_ROOT/DVSGesture           spikingjelly DVS128Gesture root
#   ncaltech     $DATA_ROOT/NCALTECH101          N-Caltech101 root
#
# The three event datasets must point at the DATASET directory, not its parent -- spikingjelly
# looks for its extracted frames there and aborts with "This dataset can not be downloaded by
# SpikingJelly" if given the parent. On first use it builds a frames cache under that directory,
# which takes a while; later runs reuse it.
#
# WHAT IS BEING MEASURED. Each model is an SNN-MoE (4 experts per block, top-1 routing) fine-
# tuned at its deployed operating point. At inference every expert is skipped for the inputs
# whose routed-token entropy falls below that expert's threshold, so the reported "active" is
# an average over the test set and is normally well below 4.
# ============================================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ---- the only thing you need to set ---------------------------------------------------------
DATA_ROOT=${DATA_ROOT:-/scratch1/bkrhee/data}

# ---- optional: environment + per-dataset path overrides -------------------------------------
VENV=${VENV:-}                       # e.g. /path/to/.venv  (leave empty to use the current env)
[ -n "$VENV" ] && [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

DATA_CIFAR10=${DATA_CIFAR10:-$DATA_ROOT}
DATA_CIFAR100=${DATA_CIFAR100:-$DATA_ROOT}
DATA_IMAGENET=${DATA_IMAGENET:-$DATA_ROOT/imagenet}
DATA_CIFAR10DVS=${DATA_CIFAR10DVS:-$DATA_ROOT/cifar10-dvs}
DATA_GESTURE=${DATA_GESTURE:-$DATA_ROOT/DVSGesture}
DATA_NCALTECH=${DATA_NCALTECH:-$DATA_ROOT/NCALTECH101}

DATASETS=${DATASETS:-"cifar10 cifar100 cifar10dvs gesture ncaltech imagenet"}
GPU=${GPU:-0}
LOGDIR=${LOGDIR:-$HERE/logs}
mkdir -p "$LOGDIR"

data_for() {
  case "$1" in
    cifar10)    echo "$DATA_CIFAR10" ;;   cifar100)  echo "$DATA_CIFAR100" ;;
    imagenet)   echo "$DATA_IMAGENET" ;;  cifar10dvs) echo "$DATA_CIFAR10DVS" ;;
    gesture)    echo "$DATA_GESTURE" ;;   ncaltech)  echo "$DATA_NCALTECH" ;;
  esac
}
field() { python3 -c "import json,sys;print(json.load(open('manifest.json'))['$1'].get('$2',''))"; }

# The two harnesses print top-1 differently: firing_num.py as timm's running "Acc@1: x (avg)",
# alloc_dvs.py as a single "top-1: x". Same for the active-expert count.
acc_of() {
  grep -oE "^top-1: [0-9.]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$" && return
  grep -oE "Acc@1: +[0-9.]+ \([0-9.]+\)" "$1" 2>/dev/null | tail -1 \
    | grep -oE "\([0-9.]+\)$" | tr -d '()'
}
act_of() {
  grep -oE "active experts \(t_e>0\): [0-9.]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$"
}

command -v python >/dev/null || { echo "!! no python on PATH (set VENV= at the top)"; exit 1; }
[ -f manifest.json ] || { echo "!! manifest.json not found -- run from the bundle directory"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || echo "!! warning: no GPU detected; this will be very slow on CPU"

echo "DATA_ROOT = $DATA_ROOT"
echo

for ds in $DATASETS; do
  CFG=$(field "$ds" config); CK=$(field "$ds" file)
  THR=$(field "$ds" exit_threshold_per_expert); EXTRA=$(field "$ds" extra)
  EVAL=$(field "$ds" eval); EVAL=${EVAL:-firing}; [ -n "${FORCE_EVAL:-}" ] && EVAL=$FORCE_EVAL
  # The batch size the reported number was measured at -- NOT always the config's
  # val_batch_size (cifar10 was measured at 64, its config says 32). Batch shape changes
  # cuDNN's algorithm choice, which moves top-1 by an image or two. Note firing_num.py's
  # DVS branch batches with args.batch_size, so this flag only bites on the static sets.
  VBS=$(field "$ds" val_batch_size)
  DDIR=$(data_for "$ds")

  echo "############################################################ $ds  (eval: $EVAL)"
  [ -f "$CFG" ] || { echo "  !! missing config $CFG"; continue; }
  [ -f "$CK" ]  || { echo "  !! missing checkpoint $CK"; continue; }
  [ -d "$DDIR" ] || { echo "  !! data directory not found: $DDIR   (set DATA_ROOT or DATA_${ds^^})"; continue; }
  echo "  data = $DDIR"

  # '--flag=value' form: some threshold lists start with -1.0000 (meaning KEEP, never prune),
  # which argparse would otherwise read as the start of a new option.
  if [ "$EVAL" = alloc ]; then
    # The calibration/eval harness, kept as a cross-check: it builds the same shuffle=False
    # DataLoader firing_num.py's DVS branch does, so the two should agree. It differs in how it
    # averages the active-expert count (unweighted over batches, so a ragged last batch counts
    # as much as a full one), which is worth a few hundredths on N-Caltech101's 914 images.
    CUDA_VISIBLE_DEVICES=$GPU python alloc_dvs.py --mode eval \
        -c "$CFG" --resume "$CK" --use-ema --data-dir "$DDIR" ${VBS:+--batch-size $VBS} \
        "--exit-threshold-per-expert=$THR" \
        > "$LOGDIR/$ds.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$GPU WORLD_SIZE=1 python firing_num.py \
        -c "$CFG" --model sdt --spike-mode lif --mlp-ratio 1 \
        -data-dir "$DDIR" --resume "$CK" --no-resume-opt \
        ${VBS:+--val-batch-size $VBS} \
        --early-exit --exit-metric input_entropy --prune-tmean --prune-only \
        "--exit-threshold-per-expert=$THR" $EXTRA \
        > "$LOGDIR/$ds.log" 2>&1
  fi
  rc=$?
  acc=$(acc_of "$LOGDIR/$ds.log")
  [ -n "$acc" ] || echo "  !! no accuracy parsed (rc=$rc) -- see $LOGDIR/$ds.log"
  echo
done

# ---------------------------------- summary ---------------------------------------------------
echo "=================== SDT-MoE inference results ==================="
printf "%-12s %-10s %-10s %-10s %-10s %s\n" dataset top-1 expected active expected status
for ds in $DATASETS; do
  L="$LOGDIR/$ds.log"
  acc=$(acc_of "$L"); act=$(act_of "$L")
  exp=$(field "$ds" ft); eact=$(field "$ds" active_post)
  st="-"
  [ -n "$acc" ] && st=$(python3 -c "
d=abs($acc-$exp); print('OK' if d<=0.15 else f'DIFFERS by {d:.2f}')" 2>/dev/null)
  printf "%-12s %-10s %-10s %-10s %-10s %s\n" "$ds" "${acc:-FAILED}" "$exp" "${act:-?}" "$eact" "$st"
done
echo
echo "'expected' is the accuracy and mean active-expert count measured on our side."
echo "A small difference is normal (fp16 vs fp32 changes LIF spike timing slightly);"
echo "a large one usually means the data directory is not the expected layout."
echo "Logs: $LOGDIR/"
