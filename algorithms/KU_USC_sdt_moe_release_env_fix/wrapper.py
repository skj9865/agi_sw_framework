"""Wrapper for KU-USC SDT-MoE (Spike-Driven Transformer with Mixture-of-Experts).

Uses the env_fix release which bundles sj_compat.py — a compatibility layer that
lets the model run on older environments (torch 2.0, timm 0.5.4, spikingjelly 0.0.0.0.12)
without CuPy or spikingjelly.activation_based.

Key integration points
----------------------
* sj_compat  – provides LIFNode/ParametricLIFNode shims, functional.reset_net,
  clean_state_dict, torch_load, and float32-casted DVS datasets.  No CuPy patching
  or spikingjelly.activation_based required.
* Module isolation – SDT-MoE ships local packages named ``model`` / ``module`` /
  ``dvs_utils`` that clash with ku_multimodal's ``model`` package.  We import
  them inside a scope that temporarily clears and restores ``sys.modules``.
"""

import sys
import os
import json
import yaml
import logging
from contextlib import suppress, contextmanager

import matplotlib
matplotlib.use("Agg")

_WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
_SDT_MOE_DIR = os.path.join(_WRAPPER_DIR, "sdt_moe_release")

# ---------------------------------------------------------------------------
# Module-isolated imports of SDT-MoE packages.  After the context manager exits
# the class/function objects remain alive via the local references below, while
# sys.modules is restored so ku_multimodal's ``model`` package is not poisoned.
# ---------------------------------------------------------------------------

@contextmanager
def _sdt_moe_import_scope():
    _conflicting = ("model", "module", "dvs_utils", "sj_compat")
    saved = {}
    for name in list(sys.modules.keys()):
        if name.split(".")[0] in _conflicting:
            saved[name] = sys.modules.pop(name)
    sys.path.insert(0, _SDT_MOE_DIR)
    try:
        yield
    finally:
        for name in list(sys.modules.keys()):
            if name.split(".")[0] in _conflicting:
                del sys.modules[name]
        sys.modules.update(saved)
        try:
            sys.path.remove(_SDT_MOE_DIR)
        except ValueError:
            pass

with _sdt_moe_import_scope():
    from sj_compat import (
        functional as _sj_functional,
        clean_state_dict as _sj_clean_state_dict,
        torch_load as _sj_torch_load,
        compat_report as _sj_compat_report,
        CIFAR10DVS as _CIFAR10DVS,
        DVS128Gesture as _DVS128Gesture,
        NCaltech101 as _NCaltech101,
    )
    from model.spikeformer import SpikeDrivenTransformer as _SpikeDrivenTransformer
    from module.ms_conv import EarlyExitMeter as _EarlyExitMeter
    from module.ms_conv import EntropyProbe as _EntropyProbe
    import dvs_utils as _dvs_utils

from core.base_algorithm import BaseAlgorithm
from core.registry import register_algorithm

_logger = logging.getLogger("ku_usc_sdt_moe")

_DATA_SUBDIRS = {
    "cifar10": "",
    "cifar100": "",
    "imagenet": "imagenet",
    "cifar10dvs": "cifar10-dvs",
    "gesture": "DVSGesture",
    "ncaltech": "NCALTECH101",
}


@register_algorithm
class KUUSCSdtMoeAlgorithm(BaseAlgorithm):

    def __init__(self):
        self._config = {}
        self._device = "cuda:0"
        self._seed = 42
        self._dataset_key = "cifar10"
        self._data_dir = None
        self._batch_size = None
        self._workers = 4

    def name(self) -> str:
        return "ku_usc_sdt_moe"

    def configure(self, config: dict) -> None:
        self._config = config
        self._device = config.get("device", "cuda:0")
        self._seed = config.get("seed", 42)
        ds = config.get("dataset", config.get("default_dataset", "cifar10"))
        norm = {"cifar10-dvs": "cifar10dvs", "ncaltech101": "ncaltech"}
        self._dataset_key = norm.get(ds, ds)
        self._data_dir = config.get("data_dir")
        self._batch_size = config.get("batch_size")
        self._workers = config.get("workers", 4)

        project_root = os.path.dirname(os.path.dirname(_WRAPPER_DIR))
        if self._data_dir and not os.path.isabs(self._data_dir):
            self._data_dir = os.path.join(project_root, self._data_dir)

    def get_supported_datasets(self) -> list:
        return list(_DATA_SUBDIRS.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_manifest(self):
        with open(os.path.join(_SDT_MOE_DIR, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        if self._dataset_key not in manifest:
            raise ValueError(
                f"Dataset '{self._dataset_key}' not in manifest.json. "
                f"Available: {list(manifest.keys())}"
            )
        return manifest[self._dataset_key]

    def _load_yaml_config(self, config_relpath):
        with open(os.path.join(_SDT_MOE_DIR, config_relpath), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return {k.replace("-", "_"): v for k, v in cfg.items()}

    def _resolve_data_dir(self):
        base = self._data_dir
        if not base:
            project_root = os.path.dirname(os.path.dirname(_WRAPPER_DIR))
            base = os.path.join(project_root, "dataset")
        subdir = _DATA_SUBDIRS.get(self._dataset_key, "")
        return os.path.join(base, subdir) if subdir else base

    def _build_model(self, yml, entry):
        import inspect

        dataset_name = yml.get("dataset", entry["dataset"])
        dvs_mode = dataset_name in ("cifar10-dvs", "cifar10-dvs-tet", "ncaltech101")

        kw = dict(
            T=yml.get("time_steps", 4),
            pretrained=yml.get("pretrained", False),
            drop_rate=yml.get("drop", 0.0),
            drop_path_rate=yml.get("drop_path", 0.2),
            drop_block_rate=yml.get("drop_block", None),
            num_heads=yml.get("num_heads", 8),
            num_classes=yml.get("num_classes", 10),
            pooling_stat=yml.get("pooling_stat", "1111"),
            img_size_h=yml.get("img_size", 32),
            img_size_w=yml.get("img_size", 32),
            patch_size=yml.get("patch_size", None),
            embed_dims=entry["dim"],
            mlp_ratios=entry.get("mlp_ratio", 1),
            in_channels=yml.get("in_channels", 3),
            qkv_bias=False,
            depths=entry["layer"],
            sr_ratios=1,
            spike_mode=yml.get("spike_mode", "lif"),
            dvs_mode=dvs_mode,
            TET=yml.get("TET", False),
            mixing_mode=yml.get("mixing_mode", "none"),
            mix_ratio=yml.get("mix_ratio", 0.5),
            moe_type=yml.get("moe_type", "allrouted"),
            num_experts=yml.get("num_experts", 4),
            top_k=yml.get("top_k", None),
            sample_routing=yml.get("sample_routing", False),
            use_ste=yml.get("use_ste", False),
            use_output_lif=yml.get("use_output_lif", False),
            alternating_moe=yml.get("alternating_moe", False),
            mlp_ratio_plain=yml.get(
                "mlp_ratio_plain", entry.get("mlp_ratio", 1)
            ),
            early_exit=True,
            exit_threshold=yml.get("exit_threshold", 0.5),
            exit_low_T=yml.get("exit_low_t", 1),
            prune_threshold=yml.get("prune_threshold", None),
            entropy_norm=not yml.get("no_entropy_norm", False),
            exit_metric="input_entropy",
            exit_mode=yml.get("exit_mode", "absolute"),
        )

        valid = set(
            inspect.signature(_SpikeDrivenTransformer.__init__).parameters.keys()
        )
        kw = {k: v for k, v in kw.items() if k in valid and v is not None}
        return _SpikeDrivenTransformer(**kw)

    def _setup_pruning(self, model, entry):
        moe_blocks = [
            blk for blk in model.block
            if hasattr(blk.mlp, "exit_threshold")
        ]
        for blk in moe_blocks:
            blk.mlp.prune_only = True
            blk.mlp.prune_tmean = True

        thr_str = entry.get("exit_threshold_per_expert", "")
        if thr_str and moe_blocks:
            flat = [float(x.strip()) for x in thr_str.split(",")]
            nblk = len(moe_blocks)
            if len(flat) % nblk == 0:
                E = len(flat) // nblk
                for bi, blk in enumerate(moe_blocks):
                    blk.mlp.exit_threshold_per_expert = flat[
                        bi * E : (bi + 1) * E
                    ]
            else:
                _logger.warning(
                    "exit_threshold_per_expert length %d not divisible by %d blocks",
                    len(flat), nblk,
                )

    def _load_checkpoint(self, model, ckpt_path):
        import torch

        if not os.path.exists(ckpt_path):
            _logger.warning("Checkpoint not found: %s", ckpt_path)
            return

        ckpt = _sj_torch_load(ckpt_path, map_location="cpu")
        if "state_dict_ema" in ckpt and not ckpt.get("ema_extracted", False):
            sd = _sj_clean_state_dict(ckpt["state_dict_ema"])
            _logger.info("Loaded EMA weights for inference")
        else:
            sd = ckpt.get("state_dict", ckpt.get("model", ckpt))

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            _logger.warning("Missing keys: %s", missing)
        if unexpected:
            _logger.warning("Unexpected keys: %s", unexpected)

    def _build_eval_loader(self, entry, yml, model):
        import torch

        dataset_name = yml.get("dataset", entry["dataset"])
        data_dir = self._resolve_data_dir()
        time_steps = yml.get("time_steps", 4)
        batch_size = self._batch_size or entry.get("val_batch_size", 64)
        workers = self._workers
        extra = entry.get("extra", "")

        if dataset_name == "cifar10-dvs":
            ds = _CIFAR10DVS(
                data_dir,
                data_type="frame",
                frames_number=time_steps,
                split_by="number",
                transform=_dvs_utils.Resize(64),
            )
            _, dataset_eval = _dvs_utils.split_to_train_test_set(0.9, ds, 10)
        elif dataset_name == "ncaltech101":
            _, dataset_eval = _dvs_utils.build_ncaltech(data_dir, True)
        elif dataset_name == "gesture":
            dataset_eval = _DVS128Gesture(
                data_dir,
                train=False,
                data_type="frame",
                frames_number=time_steps,
                split_by="number",
            )
        else:
            from timm.data import create_dataset

            dataset_eval = create_dataset(
                dataset_name,
                root=data_dir,
                split="validation",
                is_training=False,
                batch_size=batch_size,
            )

        if dataset_name in _dvs_utils.DVS_DATASET:
            loader = torch.utils.data.DataLoader(
                dataset_eval,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
            )
        else:
            from timm.data import create_loader

            img_size = yml.get("img_size", 32)
            in_channels = yml.get("in_channels", 3)
            use_prefetcher = (
                "--no-prefetcher" not in extra and torch.cuda.is_available()
            )

            loader = create_loader(
                dataset_eval,
                input_size=(in_channels, img_size, img_size),
                batch_size=batch_size,
                is_training=False,
                use_prefetcher=use_prefetcher,
                interpolation=yml.get("interpolation", "bicubic"),
                mean=tuple(yml.get("mean", [0.4914, 0.4822, 0.4465])),
                std=tuple(yml.get("std", [0.2470, 0.2435, 0.2616])),
                num_workers=workers,
                distributed=False,
                crop_pct=yml.get("crop_pct", 1.0),
                pin_memory=False,
            )

        return loader, dataset_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, **kwargs) -> dict:
        raise NotImplementedError(
            "KUUSCSdtMoeAlgorithm is an inference-only wrapper. "
            "Training is not supported through this interface."
        )

    def evaluate(self, **kwargs) -> dict:
        import time as _time
        import torch
        import torch.nn as nn
        import numpy as np
        import random as rd
        from timm.utils import AverageMeter, accuracy

        _logger.info("sj_compat: %s", _sj_compat_report())

        original_dir = os.getcwd()
        os.chdir(_SDT_MOE_DIR)

        try:
            entry = self._load_manifest()
            yml = self._load_yaml_config(entry["config"])

            device = torch.device(
                self._device if torch.cuda.is_available() else "cpu"
            )

            torch.manual_seed(self._seed)
            np.random.seed(self._seed)
            rd.seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._seed)

            model = self._build_model(yml, entry)
            self._setup_pruning(model, entry)
            self._load_checkpoint(
                model, os.path.join(_SDT_MOE_DIR, entry["file"])
            )
            model.to(device)
            model.eval()

            extra = entry.get("extra", "")
            amp_autocast = suppress
            if "--amp" in extra and torch.cuda.is_available():
                amp_autocast = torch.cuda.amp.autocast

            loader, dataset_name = self._build_eval_loader(entry, yml, model)
            loss_fn = nn.CrossEntropyLoss().to(device)

            use_prefetcher = (
                dataset_name not in _dvs_utils.DVS_DATASET
                and "--no-prefetcher" not in extra
                and torch.cuda.is_available()
            )

            losses_m = AverageMeter()
            top1_m = AverageMeter()
            top5_m = AverageMeter()
            batch_time_m = AverageMeter()
            ee_meter = _EarlyExitMeter()
            ent_probe = _EntropyProbe()

            end = _time.time()
            with torch.no_grad():
                for batch_idx, (inp, target) in enumerate(loader):
                    if not use_prefetcher:
                        inp = inp.to(device)
                    target = target.to(device)

                    with amp_autocast():
                        output, _ = model(inp, hook=dict())

                    ee_meter.update(model)
                    ent_probe.update(model)

                    loss = loss_fn(output, target)
                    _sj_functional.reset_net(model)

                    acc1, acc5 = accuracy(output, target, topk=(1, 5))

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    losses_m.update(loss.item(), inp.size(0))
                    top1_m.update(acc1.item(), output.size(0))
                    top5_m.update(acc5.item(), output.size(0))
                    batch_time_m.update(_time.time() - end)
                    end = _time.time()

                    if batch_idx % 100 == 0:
                        _logger.info(
                            "Test: [%4d/%d]  Time: %.3f (%.3f)  "
                            "Loss: %.4f (%.4f)  "
                            "Acc@1: %.4f (%.4f)  Acc@5: %.4f (%.4f)",
                            batch_idx,
                            len(loader),
                            batch_time_m.val,
                            batch_time_m.avg,
                            losses_m.val,
                            losses_m.avg,
                            top1_m.val,
                            top1_m.avg,
                            top5_m.val,
                            top5_m.avg,
                        )

            ee_meter.report()

            return {
                "accuracy": top1_m.avg / 100.0,
                "top1": top1_m.avg,
                "top5": top5_m.avg,
                "loss": losses_m.avg,
                "dataset": self._dataset_key,
            }
        finally:
            os.chdir(original_dir)
