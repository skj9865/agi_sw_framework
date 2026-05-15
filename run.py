#!/usr/bin/env python
"""Unified SW Framework - CLI entry point."""

import argparse
import os
import sys

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import core (this also triggers algorithm registration via algorithms/__init__.py)
from core.runner import Runner
import algorithms  # noqa: F401 - triggers @register_algorithm decorators


def main():
    parser = argparse.ArgumentParser(
        description="Unified SW Framework - Brain-inspired Algorithm Benchmark"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all registered algorithms"
    )
    parser.add_argument(
        "--algorithm", "-a", type=str, default=None,
        help="Algorithm to run (e.g. 'ff', 'monty')"
    )
    parser.add_argument(
        "--compare", nargs="+", type=str, default=None,
        help="Compare multiple algorithms (e.g. --compare ff monty)"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default=None,
        help="Dataset to use (e.g. 'cifar10', 'mnist', 'svhn')"
    )
    parser.add_argument(
        "--mode", "-m", type=str, default="train",
        choices=["train", "evaluate"],
        help="Run mode: 'train' or 'evaluate' (default: train)"
    )
    parser.add_argument(
        "--config", "-c", type=str,
        default=os.path.join(ROOT_DIR, "config", "framework_config.yaml"),
        help="Path to framework config YAML"
    )

    args = parser.parse_args()

    if args.list:
        Runner.list_algorithms()
        return

    runner = Runner(config_path=args.config)

    if args.compare:
        runner.compare(args.compare, dataset=args.dataset, mode=args.mode)
        return

    if not args.algorithm:
        parser.print_help()
        print("\nError: --algorithm or --compare is required (or use --list)")
        sys.exit(1)

    result = runner.run(args.algorithm, dataset=args.dataset, mode=args.mode)

    print(f"\n{'='*50}")
    print(f"Final Result: {result}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
