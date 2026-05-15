import os
import yaml
import time
from core.registry import AlgorithmRegistry


class Runner:
    def __init__(self, config_path: str = None):
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

    def _get_framework_config(self) -> dict:
        return self.config.get("framework", {})

    def _get_algorithm_config(self, algo_name: str) -> dict:
        return self.config.get("algorithms", {}).get(algo_name, {})

    def run(self, algo_name: str, dataset: str = None, mode: str = "train") -> dict:
        """Run a single algorithm in train or evaluate mode."""
        algo_cls = AlgorithmRegistry.get(algo_name)
        algo = algo_cls()

        # Merge framework config with algorithm-specific config
        fw_config = self._get_framework_config()
        algo_config = self._get_algorithm_config(algo_name)
        merged = {**fw_config, **algo_config}
        if dataset:
            merged["dataset"] = dataset

        algo.configure(merged)

        print(f"[Framework] Running '{algo_name}' | dataset={dataset} | mode={mode}")
        start = time.time()

        if mode == "train":
            result = algo.train()
        elif mode == "evaluate":
            result = algo.evaluate()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'evaluate'.")

        elapsed = time.time() - start
        result["elapsed_sec"] = round(elapsed, 2)
        print(f"[Framework] Done in {elapsed:.2f}s | result: {result}")
        return result

    def compare(self, algo_names: list, dataset: str = None, mode: str = "evaluate") -> dict:
        """Run multiple algorithms and compare results."""
        results = {}
        for name in algo_names:
            print(f"\n{'='*60}")
            print(f"  [{name.upper()}]")
            print(f"{'='*60}")
            try:
                results[name] = self.run(name, dataset=dataset, mode=mode)
            except Exception as e:
                print(f"[Framework] '{name}' failed: {e}")
                results[name] = {"error": str(e)}

        # Print comparison table
        print(f"\n{'='*60}")
        print(f"  COMPARISON RESULTS ({mode})")
        print(f"{'='*60}")
        print(f"{'Algorithm':<15s} {'Accuracy':>10s} {'Time (s)':>10s} {'Dataset':<15s}")
        print(f"{'-'*50}")
        for name, r in results.items():
            if "error" in r:
                print(f"{name:<15s} {'FAILED':>10s}")
            else:
                acc = r.get("accuracy", "N/A")
                acc_str = f"{acc*100:.2f}%" if isinstance(acc, float) else str(acc)
                elapsed = r.get("elapsed_sec", "N/A")
                ds = r.get("dataset", "N/A")
                print(f"{name:<15s} {acc_str:>10s} {elapsed:>10} {ds:<15s}")
        print(f"{'='*60}")

        return results

    @staticmethod
    def list_algorithms():
        algos = AlgorithmRegistry.list_all()
        if not algos:
            print("[Framework] No algorithms registered.")
            return
        print("[Framework] Registered algorithms:")
        for name in algos:
            algo_cls = AlgorithmRegistry.get(name)
            algo = algo_cls()
            datasets = algo.get_supported_datasets()
            print(f"  - {name:20s} datasets: {', '.join(datasets)}")
