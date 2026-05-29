"""Training and evaluation functions extracted from notebook."""

import time
import random
import logging
from collections import OrderedDict
from contextlib import suppress

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from spikingjelly.clock_driven import functional
from timm.utils import AverageMeter, accuracy, dispatch_clip_grad
from timm.models import model_parameters

_logger = logging.getLogger("ku_multimodal")


# ---------------------------------------------------------------------------
# LabelMatchedPairDataset (from notebook Cell 9 - exact copy)
# ---------------------------------------------------------------------------

def _coerce_dataset(ds_or_loader):
    if isinstance(ds_or_loader, DataLoader):
        return ds_or_loader.dataset
    return ds_or_loader


def _enumerate_labels(ds, map_fn, num_classes=10):
    N = len(ds)
    labels = [None] * N
    buckets = [[] for _ in range(num_classes)]
    for i in range(N):
        sample = ds[i]
        if isinstance(sample, dict):
            y = sample.get("label")
        else:
            _, y = sample
        if torch.is_tensor(y):
            y = int(y.item())
        else:
            y = int(y)
        d = int(map_fn(y))
        labels[i] = d
        if 0 <= d < num_classes:
            buckets[d].append(i)
    return labels, buckets


class LabelMatchedPairDataset(Dataset):
    """Pairs items from two datasets by digit label.

    ds_a: anchor (e.g. SHD), ds_b: secondary (e.g. MNIST).
    map_a/map_b: map raw label -> digit for bucketing.
    """

    def __init__(self, ds_a, ds_b, map_a=lambda y: y, map_b=lambda y: y,
                 num_classes=10, drop_missing=False, randomized=True, seed=42):
        self.ds_a = _coerce_dataset(ds_a)
        self.ds_b = _coerce_dataset(ds_b)
        self.num_classes = num_classes
        self.randomized = randomized
        self.rng = random.Random(seed)

        self.labels_a, _ = _enumerate_labels(self.ds_a, map_a, num_classes)
        _, self.buckets_b = _enumerate_labels(self.ds_b, map_b, num_classes)

        present = {d for d in range(num_classes) if len(self.buckets_b[d]) > 0}
        if not present:
            raise ValueError("Secondary dataset has no digits after mapping.")
        if drop_missing:
            self.idx_a = [i for i, d in enumerate(self.labels_a) if d in present]
        else:
            self.idx_a = list(range(len(self.ds_a)))
            missing = sorted({d for d in set(self.labels_a) if d not in present})
            if missing:
                raise ValueError(f"No samples for digits {missing} in secondary dataset.")

        self.ptr = [0] * num_classes
        for d in range(num_classes):
            self.rng.shuffle(self.buckets_b[d])

    def __len__(self):
        return len(self.idx_a)

    def _pick_b_index(self, digit):
        bucket = self.buckets_b[digit]
        if self.randomized:
            return self.rng.choice(bucket)
        j = self.ptr[digit]
        self.ptr[digit] = (j + 1) % len(bucket)
        return bucket[j]

    def __getitem__(self, i):
        ai = self.idx_a[i]
        x_aud, y_b_raw = self.ds_a[ai]
        digit = self.labels_a[ai]

        bj = self._pick_b_index(digit)
        x_img, y_a_raw = self.ds_b[bj]

        return {
            "image": x_img,
            "audio": x_aud,
            "label": int(digit),
        }


# ---------------------------------------------------------------------------
# DualTimmConcat
# ---------------------------------------------------------------------------

class DualTimmConcat(nn.Module):
    """Two SNN backbones with concatenated feature fusion."""

    def __init__(self, model_shd, model_mnist, feat_dim_shd, feat_dim_mnist,
                 num_classes=20, dropout=0.0):
        super().__init__()
        self.m1 = model_mnist
        self.m2 = model_shd
        self.out_dim = feat_dim_mnist + feat_dim_shd
        self.out_dim2 = self.out_dim // 2
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(self.out_dim, self.out_dim2)
        self.head2 = nn.Linear(self.out_dim2, num_classes)
        self.num_classes = num_classes

    def forward(self, x_mnist, x_shd):
        z1 = self.m1(x_mnist)
        z2 = self.m2(x_shd)
        z = torch.cat([z1, z2], dim=-1)
        z = self.dropout(z)
        z = self.head(z)
        z = self.head2(z)
        return z


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------

def train_one_epoch(epoch, model, loader, optimizer, loss_fn,
                    lr_scheduler=None, amp_autocast=suppress,
                    loss_scaler=None, clip_grad=None, clip_mode="norm",
                    log_interval=50, device="cuda"):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    end = time.time()
    num_updates = epoch * len(loader)

    for batch_idx, y in enumerate(loader):
        image = y["image"].to(device)
        audio = y["audio"].to(device)
        target = y["label"].to(device)

        with amp_autocast:
            output = model(image, audio)
            loss = loss_fn(output, target)

        losses_m.update(loss.item(), image.size(0))
        optimizer.zero_grad()

        if loss_scaler is not None:
            loss_scaler(loss, optimizer,
                        clip_grad=clip_grad, clip_mode=clip_mode,
                        parameters=model_parameters(model))
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model_parameters(model),
                                   value=clip_grad, mode=clip_mode)
            optimizer.step()

        functional.reset_net(model)

        num_updates += 1
        batch_time_m.update(time.time() - end)

        if batch_idx % log_interval == 0:
            _logger.info(
                f"Train: {epoch} [{batch_idx:>4d}/{len(loader)}]  "
                f"Loss: {losses_m.val:>9.6f} ({losses_m.avg:>6.4f})  "
                f"Time: {batch_time_m.val:.3f}s"
            )

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    return OrderedDict([("loss", losses_m.avg)])


# ---------------------------------------------------------------------------
# validate_multimodal
# ---------------------------------------------------------------------------

def validate_multimodal(model, loader, loss_fn, amp_autocast=suppress, device="cuda"):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()
    end = time.time()

    with torch.no_grad():
        for batch_idx, y in enumerate(loader):
            image = y["image"].to(device)
            audio = y["audio"].to(device)
            target = y["label"].to(device)

            with amp_autocast:
                output = model(image, audio)

            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses_m.update(loss.item(), image.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()

    metrics = OrderedDict([
        ("loss", losses_m.avg),
        ("top1", top1_m.avg),
        ("top5", top5_m.avg),
    ])
    _logger.info(
        f"Eval: Loss: {losses_m.avg:.4f}  "
        f"Acc@1: {top1_m.avg:.2f}  Acc@5: {top5_m.avg:.2f}"
    )
    return metrics
