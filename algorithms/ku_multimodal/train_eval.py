"""Training and evaluation functions extracted from notebook."""

import time
import logging
import random
from collections import OrderedDict
from contextlib import suppress

import torch
import torch.nn as nn
import torchvision.utils
from torch.utils.data import Dataset, DataLoader

from spikingjelly.clock_driven import functional
from timm.utils import AverageMeter, accuracy, dispatch_clip_grad
from timm.models import model_parameters

_logger = logging.getLogger("ku_multimodal")


# ---------------------------------------------------------------------------
# LabelMatchedPairDataset
# ---------------------------------------------------------------------------

class LabelMatchedPairDataset(Dataset):
    """Pair samples from two datasets by matching labels."""

    def __init__(self, ds_a, ds_b, map_a=None, map_b=None,
                 num_classes=20, randomized=True, seed=123):
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.map_a = map_a or (lambda y: y)
        self.map_b = map_b or (lambda y: y)
        self.num_classes = num_classes
        self.randomized = randomized
        self.rng = random.Random(seed)

        # Build label -> index mapping for ds_b
        self.b_by_label = {}
        for idx in range(len(ds_b)):
            _, y = ds_b[idx]
            y_mapped = y % 10  # map to 0-9
            self.b_by_label.setdefault(y_mapped, []).append(idx)

        self.b_counters = {k: 0 for k in self.b_by_label}

    def __len__(self):
        return len(self.ds_a)

    def __getitem__(self, idx):
        x_a, y_a = self.ds_a[idx]
        label_a = int(y_a)
        # Map SHD label (0-19) to MNIST label (0-9) for matching
        match_key = label_a % 10

        candidates = self.b_by_label.get(match_key, [])
        if not candidates:
            b_idx = 0
        elif self.randomized:
            b_idx = self.rng.choice(candidates)
        else:
            c = self.b_counters[match_key]
            b_idx = candidates[c % len(candidates)]
            self.b_counters[match_key] = c + 1

        x_b, y_b = self.ds_b[b_idx]
        combined_label = self.map_b(int(y_b))

        # Ensure tensors
        if not isinstance(x_a, torch.Tensor):
            x_a = torch.tensor(x_a, dtype=torch.float32)
        if not isinstance(x_b, torch.Tensor):
            x_b = torch.tensor(x_b, dtype=torch.float32)

        return {"audio": x_a, "image": x_b, "label": combined_label}


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

        with amp_autocast():
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

            with amp_autocast():
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
