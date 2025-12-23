#!/usr/bin/env python3
"""Benchmark XGBoost and CatBoost with CPU and GPU configurations."""

import argparse
import time
import gc
import subprocess
import threading
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import xgboost as xgb
import catboost as cb


class GPUMonitor:
    """Monitor GPU utilization during training."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.running = False
        self.thread = None
        self.gpu_utils = []
        self.gpu_mem = []

    def _monitor(self):
        while self.running:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    util_snapshot = {}
                    mem_snapshot = {}
                    for line in result.stdout.strip().split("\n"):
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            gpu_id = int(parts[0])
                            util_snapshot[gpu_id] = int(parts[1])
                            mem_snapshot[gpu_id] = int(parts[2])
                    self.gpu_utils.append(util_snapshot)
                    self.gpu_mem.append(mem_snapshot)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.gpu_utils = []
        self.gpu_mem = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def get_stats(self):
        if not self.gpu_utils:
            return {}
        all_gpus = set()
        for snapshot in self.gpu_utils:
            all_gpus.update(snapshot.keys())
        stats = {}
        for gpu_id in sorted(all_gpus):
            utils = [s.get(gpu_id, 0) for s in self.gpu_utils]
            mems = [s.get(gpu_id, 0) for s in self.gpu_mem]
            stats[gpu_id] = {
                "avg_util": np.mean(utils) if utils else 0,
                "max_util": max(utils) if utils else 0,
                "avg_mem": np.mean(mems) if mems else 0,
                "max_mem": max(mems) if mems else 0,
            }
        return stats


def generate_dataset(n_samples: int, n_features: int, random_state: int = 42):
    """Generate a synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=n_features // 4,
        n_clusters_per_class=2,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def benchmark_xgboost(X_train, y_train, X_test, y_test, device: str, n_gpus: int = 1, n_rounds: int = 100):
    """Benchmark XGBoost training."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.1,
        "verbosity": 1,  # Show some output
    }

    if device == "cpu":
        params["device"] = "cpu"
        params["nthread"] = -1
    else:
        # XGBoost 2.x+ only supports single GPU per process
        # Multi-GPU requires Dask or Ray
        params["device"] = "cuda:0"
        params["tree_method"] = "hist"

    gc.collect()
    monitor = GPUMonitor()
    monitor.start()

    start = time.perf_counter()
    model = xgb.train(params, dtrain, num_boost_round=n_rounds, evals=[(dtest, "test")], verbose_eval=False)
    elapsed = time.perf_counter() - start

    monitor.stop()
    return elapsed, model, monitor.get_stats()


def benchmark_catboost(X_train, y_train, X_test, y_test, device: str, n_gpus: int = 1, n_rounds: int = 100):
    """Benchmark CatBoost training."""
    params = {
        "iterations": n_rounds,
        "depth": 6,
        "learning_rate": 0.1,
        "loss_function": "Logloss",
        "verbose": True,  # Show progress
    }

    if device == "cpu":
        params["task_type"] = "CPU"
        params["thread_count"] = -1
    else:
        params["task_type"] = "GPU"
        if n_gpus > 1:
            # Multi-GPU: specify device range
            params["devices"] = "0-" + str(n_gpus - 1)
            # Also try explicit list format
            # params["devices"] = ",".join(str(i) for i in range(n_gpus))
        else:
            params["devices"] = "0"

    model = cb.CatBoostClassifier(**params)

    gc.collect()
    monitor = GPUMonitor()
    monitor.start()

    start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    elapsed = time.perf_counter() - start

    monitor.stop()
    return elapsed, model, monitor.get_stats()


def format_gpu_stats(gpu_stats: dict) -> str:
    if not gpu_stats:
        return "N/A"
    parts = []
    for gpu_id, stats in sorted(gpu_stats.items()):
        parts.append(f"GPU{gpu_id}: {stats['avg_util']:.0f}%/{stats['max_util']}%")
    return " | ".join(parts)


def run_single_test(library: str, device: str, n_gpus: int, n_samples: int, n_features: int, n_rounds: int):
    """Run a single benchmark test."""
    print(f"\n{'='*80}")
    print(f"Testing: {library.upper()} | Device: {device} | GPUs: {n_gpus}")
    print(f"Dataset: {n_samples:,} samples x {n_features} features | Rounds: {n_rounds}")
    print("=" * 80)

    print("Generating dataset...", end=" ", flush=True)
    X_train, X_test, y_train, y_test = generate_dataset(n_samples, n_features)
    print(f"Done. Train: {X_train.shape}")

    if library == "xgboost":
        elapsed, _, gpu_stats = benchmark_xgboost(X_train, y_train, X_test, y_test, device, n_gpus, n_rounds)
    else:
        elapsed, _, gpu_stats = benchmark_catboost(X_train, y_train, X_test, y_test, device, n_gpus, n_rounds)

    print(f"\nTime: {elapsed:.2f}s")
    if device == "gpu":
        print(f"GPU Usage: {format_gpu_stats(gpu_stats)}")

    return elapsed, gpu_stats


def run_full_benchmarks(args):
    """Run full benchmark suite."""
    problem_sizes = [
        (100_000, 50),
        (500_000, 100),
        (1_000_000, 100),
        (2_000_000, 100),
    ]

    configs = []
    if not args.skip_cpu:
        configs.append(("cpu", 0, "CPU"))
    if not args.skip_1gpu:
        configs.append(("gpu", 1, "1 GPU"))
    if not args.skip_4gpu:
        configs.append(("gpu", 4, "4 GPUs"))

    libraries = []
    if not args.skip_xgboost:
        libraries.append("xgboost")
    if not args.skip_catboost:
        libraries.append("catboost")

    results = []

    print("=" * 100)
    print("XGBoost & CatBoost Benchmark")
    print("=" * 100)
    print(f"Libraries: {', '.join(libraries)}")
    print(f"Configs: {', '.join(c[2] for c in configs)}")
    print(f"Rounds: {args.rounds}")

    for n_samples, n_features in problem_sizes:
        print(f"\n{'='*100}")
        print(f"Dataset: {n_samples:,} samples x {n_features} features")
        print("=" * 100)

        X_train, X_test, y_train, y_test = generate_dataset(n_samples, n_features)

        for device, n_gpus, config_label in configs:
            print(f"\n--- {config_label} ---")

            row = {
                "Samples": f"{n_samples:,}",
                "Features": n_features,
                "Config": config_label,
            }

            for lib in libraries:
                try:
                    print(f"  {lib}: ", end="", flush=True)
                    if lib == "xgboost":
                        elapsed, _, gpu_stats = benchmark_xgboost(
                            X_train, y_train, X_test, y_test, device, n_gpus, args.rounds
                        )
                    else:
                        elapsed, _, gpu_stats = benchmark_catboost(
                            X_train, y_train, X_test, y_test, device, n_gpus, args.rounds
                        )
                    print(f"{elapsed:.2f}s")
                    if device == "gpu":
                        print(f"           GPU: {format_gpu_stats(gpu_stats)}")
                    row[f"{lib} (s)"] = f"{elapsed:.2f}"
                    row[f"{lib} GPU"] = format_gpu_stats(gpu_stats) if device == "gpu" else "-"
                except Exception as e:
                    print(f"Error - {e}")
                    row[f"{lib} (s)"] = "N/A"
                    row[f"{lib} GPU"] = "Error"

            results.append(row)

        del X_train, X_test, y_train, y_test
        gc.collect()

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    df = pd.DataFrame(results)
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))


def main():
    parser = argparse.ArgumentParser(description="Benchmark XGBoost and CatBoost")

    # Quick test mode
    parser.add_argument("--test", choices=["xgb-1gpu", "xgb-4gpu", "cb-1gpu", "cb-4gpu", "cb-cpu", "xgb-cpu"],
                        help="Run a single quick test")

    # Skip flags
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU tests")
    parser.add_argument("--skip-1gpu", action="store_true", help="Skip 1 GPU tests")
    parser.add_argument("--skip-4gpu", action="store_true", help="Skip 4 GPU tests")
    parser.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost tests")
    parser.add_argument("--skip-catboost", action="store_true", help="Skip CatBoost tests")

    # Test parameters
    parser.add_argument("--samples", type=int, default=500_000, help="Number of samples for quick test")
    parser.add_argument("--features", type=int, default=100, help="Number of features for quick test")
    parser.add_argument("--rounds", type=int, default=100, help="Number of boosting rounds")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs for multi-GPU test")

    args = parser.parse_args()

    if args.test:
        # Quick single test mode
        test_map = {
            "xgb-cpu": ("xgboost", "cpu", 0),
            "xgb-1gpu": ("xgboost", "gpu", 1),
            "xgb-4gpu": ("xgboost", "gpu", args.gpus),
            "cb-cpu": ("catboost", "cpu", 0),
            "cb-1gpu": ("catboost", "gpu", 1),
            "cb-4gpu": ("catboost", "gpu", args.gpus),
        }
        lib, device, n_gpus = test_map[args.test]
        run_single_test(lib, device, n_gpus, args.samples, args.features, args.rounds)
    else:
        # Full benchmark mode
        run_full_benchmarks(args)


if __name__ == "__main__":
    main()
