import sys
import os
import random
import logging
import yaml
from contextlib import suppress

import matplotlib
matplotlib.use("Agg")

_KU_DIR = os.path.dirname(os.path.abspath(__file__))
if _KU_DIR not in sys.path:
    sys.path.insert(0, _KU_DIR)

from core.base_algorithm import BaseAlgorithm
from core.registry import register_algorithm

_logger = logging.getLogger("ku_multimodal")


@register_algorithm
class KUMultimodalAlgorithm(BaseAlgorithm):

    def __init__(self):
        self._config = {}
        self._device = "cuda:0"
        self._seed = 42
        self._epochs = 300
        self._batch_size = 32
        self._lr = 5e-4
        self._shd_data_dir = None
        self._mnist_data_dir = None
        self._checkpoint_path = None
        self._num_classes = 20
        self._time_step = 4
        self._dim = 256
        self._num_heads = 8
        self._patch_size = 8
        self._layer = 2
        self._n_time_bins = 64
        self._n_freq_bins = 64

    def name(self) -> str:
        return "ku_multimodal"

    def configure(self, config: dict) -> None:
        self._config = config
        self._device = config.get("device", "cuda:0")
        self._seed = config.get("seed", 42)
        self._epochs = config.get("epochs", 300)
        self._batch_size = config.get("batch_size", 32)
        self._lr = float(config.get("lr", 5e-4))
        self._shd_data_dir = config.get("shd_data_dir")
        self._mnist_data_dir = config.get("mnist_data_dir")
        self._checkpoint_path = config.get("checkpoint_path")
        self._num_classes = config.get("num_classes", 20)
        self._time_step = config.get("time_step", 4)
        self._dim = config.get("dim", 256)
        self._num_heads = config.get("num_heads", 8)
        self._patch_size = config.get("patch_size", 8)
        self._layer = config.get("layer", 2)
        self._n_time_bins = config.get("n_time_bins", 64)
        self._n_freq_bins = config.get("n_freq_bins", 64)

        # Resolve relative paths against project root
        project_root = os.path.dirname(os.path.dirname(_KU_DIR))
        for attr in ("_shd_data_dir", "_mnist_data_dir", "_checkpoint_path"):
            val = getattr(self, attr)
            if val and not os.path.isabs(val):
                setattr(self, attr, os.path.join(project_root, val))

    def get_supported_datasets(self) -> list:
        return ["shd_mnist"]

    def _build_model(self):
        """Build DualTimmConcat model with two SDT backbones."""
        import torch
        from timm.models import create_model, load_checkpoint
        from train_eval import DualTimmConcat

        # Import model registration (registers 'sdt' with timm)
        import importlib
        import model as _ku_model
        importlib.reload(_ku_model)

        model_kwargs = dict(
            pretrained=False,
            drop_rate=0., drop_path_rate=0., drop_block_rate=None,
            img_size_h=self._n_freq_bins, img_size_w=self._n_freq_bins,
            patch_size=self._patch_size, embed_dims=self._dim,
            num_heads=self._num_heads, mlp_ratios=4,
            in_channels=1, num_classes=self._num_classes, qkv_bias=False,
            depths=self._layer, sr_ratios=1, T=self._time_step,
        )

        backbone_mnist = create_model("sdt", **model_kwargs)
        backbone_shd = create_model("sdt", **model_kwargs)

        feat_dim = getattr(backbone_mnist, "num_features", self._dim)

        dual_model = DualTimmConcat(
            model_shd=backbone_shd,
            model_mnist=backbone_mnist,
            feat_dim_shd=feat_dim,
            feat_dim_mnist=feat_dim,
            num_classes=self._num_classes,
        )

        if self._checkpoint_path and os.path.exists(self._checkpoint_path):
            _logger.info(f"Loading checkpoint: {self._checkpoint_path}")
            load_checkpoint(dual_model, self._checkpoint_path, strict=False)

        return dual_model

    def _build_datasets(self):
        """Build paired SHD + MNIST datasets."""
        import torch
        from torchvision import transforms
        from torchvision.datasets import MNIST
        import spiking_audio_datasets

        shd_train, shd_test = spiking_audio_datasets.get_spiking_audio_datasets_just_dataset(
            dataset="shd",
            dir=self._shd_data_dir,
            batch_size=self._batch_size,
            n_time_bins=self._n_time_bins,
            n_freq_bins=self._n_freq_bins,
            no_aug=False,
            time_jitter=1, spatial_jitter=0.55, max_drop_chunk=0.02,
            noise=35, drop_event=0.1, time_skew=1.2, cut_mix=0.3,
            sa_portion=0.15, sa_times=2,
            fp16=True, num_workers=0, pin_memory=False,
        )

        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        mnist_train = MNIST(self._mnist_data_dir, train=True, download=True, transform=mnist_transform)
        mnist_test = MNIST(self._mnist_data_dir, train=False, download=True, transform=mnist_transform)

        from train_eval import LabelMatchedPairDataset

        # SHD has 20 classes (0-19), MNIST has 10 (0-9)
        # Notebook Cell 10: map_b = y + random.randint(0,1)*10 to create 20 classes
        paired_train = LabelMatchedPairDataset(
            ds_a=shd_train, ds_b=mnist_train,
            map_a=lambda y: y,
            map_b=lambda y: y + random.randint(0, 1) * 10,
            num_classes=self._num_classes, randomized=True, seed=self._seed,
        )
        paired_test = LabelMatchedPairDataset(
            ds_a=shd_test, ds_b=mnist_test,
            map_a=lambda y: y,
            map_b=lambda y: y + random.randint(0, 1) * 10,
            num_classes=self._num_classes, randomized=True, seed=self._seed,
        )
        return paired_train, paired_test

    def train(self, **kwargs) -> dict:
        """Run training loop."""
        import torch
        from torch.utils.data import DataLoader
        from timm.optim import create_optimizer_v2
        from timm.scheduler import create_scheduler
        from timm.loss import LabelSmoothingCrossEntropy
        from timm.utils import NativeScaler
        from types import SimpleNamespace
        from train_eval import train_one_epoch, validate_multimodal

        original_dir = os.getcwd()
        os.chdir(_KU_DIR)

        try:
            torch.manual_seed(self._seed)
            device = torch.device(self._device if torch.cuda.is_available() else "cpu")

            model = self._build_model()
            model.to(device)

            paired_train, paired_test = self._build_datasets()
            loader_train = DataLoader(paired_train, batch_size=self._batch_size,
                                      shuffle=True, num_workers=0)
            loader_eval = DataLoader(paired_test, batch_size=self._batch_size,
                                     shuffle=False, num_workers=0)

            # Optimizer & scheduler
            opt_args = SimpleNamespace(
                opt="adamw", lr=self._lr, weight_decay=0.06, momentum=0.9,
            )
            optimizer = create_optimizer_v2(model, **{
                "opt": opt_args.opt, "lr": opt_args.lr,
                "weight_decay": opt_args.weight_decay,
            })

            sched_args = SimpleNamespace(
                sched="cosine", epochs=self._epochs, min_lr=1e-5,
                cooldown_epochs=10, warmup_epochs=20, warmup_lr=1e-5,
                decay_rate=0.1,
            )
            lr_scheduler, _ = create_scheduler(sched_args, optimizer)

            loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
            amp_autocast = torch.cuda.amp.autocast(dtype=torch.float16) if torch.cuda.is_available() else suppress()
            loss_scaler = NativeScaler() if torch.cuda.is_available() else None

            best_acc = 0.0
            for epoch in range(self._epochs):
                train_metrics = train_one_epoch(
                    epoch, model, loader_train, optimizer, loss_fn,
                    lr_scheduler=lr_scheduler, amp_autocast=amp_autocast,
                    loss_scaler=loss_scaler, device=device,
                )
                eval_metrics = validate_multimodal(
                    model, loader_eval, loss_fn,
                    amp_autocast=amp_autocast, device=device,
                )
                if lr_scheduler is not None:
                    lr_scheduler.step(epoch + 1, eval_metrics["top1"])

                if eval_metrics["top1"] > best_acc:
                    best_acc = eval_metrics["top1"]
                    _logger.info(f"Epoch {epoch}: new best acc {best_acc:.2f}%")

            return {
                "accuracy": best_acc / 100.0,
                "top1": best_acc,
                "dataset": "shd_mnist",
            }
        finally:
            os.chdir(original_dir)

    def evaluate(self, **kwargs) -> dict:
        """Run evaluation with pretrained checkpoint."""
        import torch
        from torch.utils.data import DataLoader
        from timm.loss import LabelSmoothingCrossEntropy
        from train_eval import validate_multimodal

        original_dir = os.getcwd()
        os.chdir(_KU_DIR)

        try:
            device = torch.device(self._device if torch.cuda.is_available() else "cpu")

            model = self._build_model()
            model.to(device)

            _, paired_test = self._build_datasets()
            loader_eval = DataLoader(paired_test, batch_size=self._batch_size,
                                     shuffle=False, num_workers=0)

            loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
            amp_autocast = torch.cuda.amp.autocast(dtype=torch.float16) if torch.cuda.is_available() else suppress()

            eval_metrics = validate_multimodal(
                model, loader_eval, loss_fn,
                amp_autocast=amp_autocast, device=device,
            )

            return {
                "accuracy": eval_metrics["top1"] / 100.0,
                "top1": eval_metrics["top1"],
                "top5": eval_metrics["top5"],
                "loss": eval_metrics["loss"],
                "dataset": "shd_mnist",
            }
        finally:
            os.chdir(original_dir)
