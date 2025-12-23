#!/usr/bin/env python3
"""
Reproduce CatBoost GPU benchmark from:
https://towardsdatascience.com/10x-times-fast-catboost-training-speed-with-an-nvidia-gpu-5ffefd9b57a6/

Original results:
- Dataset: HIGGS (10.5M samples, 28 features)
- 100 iterations: 7.6x GPU speedup
- 500 iterations: 9.8x GPU speedup
- 1000 iterations: 9.7x GPU speedup
"""

import time
import gc
import subprocess
import threading
import argparse
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import catboost as cb


class GPUMonitor:
    """Monitor GPU utilization during training."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.running = False
        self.thread = None
        self.gpu_utils = []

    def _monitor(self):
        while self.running:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    snapshot = {}
                    for line in result.stdout.strip().split("\n"):
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            gpu_id = int(parts[0])
                            snapshot[gpu_id] = {"util": int(parts[1]), "mem": int(parts[2])}
                    self.gpu_utils.append(snapshot)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.gpu_utils = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def get_summary(self):
        if not self.gpu_utils:
            return "N/A"
        all_gpus = set()
        for s in self.gpu_utils:
            all_gpus.update(s.keys())
        parts = []
        for gpu_id in sorted(all_gpus):
            utils = [s[gpu_id]["util"] for s in self.gpu_utils if gpu_id in s]
            if utils:
                parts.append(f"GPU{gpu_id}: {np.mean(utils):.0f}%/{max(utils)}%")
        return " | ".join(parts)


def generate_higgs_like_dataset(n_samples: int = 10_500_000, n_features: int = 28):
    """Generate a dataset similar to HIGGS."""
    print(f"Generating HIGGS-like dataset: {n_samples:,} samples x {n_features} features...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=3,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def benchmark_catboost(X_train, y_train, X_test, y_test, task_type: str, iterations: int,
                       depth: int = 6, devices: str = None):
    """Run CatBoost benchmark."""
    params = {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": 0.03,
        "loss_function": "Logloss",
        "task_type": task_type,
        "verbose": False,
    }

    if task_type == "GPU":
        params["devices"] = devices if devices else "0"

    model = cb.CatBoostClassifier(**params)

    gc.collect()
    monitor = GPUMonitor()
    if task_type == "GPU":
        monitor.start()

    start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    elapsed = time.perf_counter() - start

    if task_type == "GPU":
        monitor.stop()
        gpu_info = monitor.get_summary()
    else:
        gpu_info = "-"

    return elapsed, gpu_info


def main():
    parser = argparse.ArgumentParser(description="Reproduce HIGGS CatBoost benchmark")
    parser.add_argument("--samples", type=int, default=10_500_000, help="Number of samples")
    parser.add_argument("--features", type=int, default=28, help="Number of features")
    parser.add_argument("--iterations", type=int, nargs="+", default=[100, 500, 1000],
                        help="Iterations to test")
    parser.add_argument("--depth", type=int, default=6, help="Tree depth")
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU tests")
    parser.add_argument("--gpus", type=str, default="0", help="GPU devices (e.g., '0' or '0-3')")
    args = parser.parse_args()

    print("=" * 80)
    print("CatBoost GPU Benchmark - Reproducing TowardsDataScience Article")
    print("=" * 80)
    print(f"Original article: 10.5M samples, 28 features")
    print(f"This test: {args.samples:,} samples, {args.features} features")
    print(f"Tree depth: {args.depth}")
    print(f"GPU devices: {args.gpus}")
    print("=" * 80)

    # Generate dataset
    X_train, X_test, y_train, y_test = generate_higgs_like_dataset(args.samples, args.features)

    results = []

    for iterations in args.iterations:
        print(f"\n{'='*60}")
        print(f"ITERATIONS: {iterations}")
        print("=" * 60)

        # CPU benchmark
        if not args.skip_cpu:
            print(f"\nCPU training (depth={args.depth})...", end=" ", flush=True)
            cpu_time, _ = benchmark_catboost(X_train, y_train, X_test, y_test, "CPU", iterations, args.depth)
            print(f"{cpu_time:.2f}s")
        else:
            cpu_time = None
            print("CPU skipped")

        # GPU benchmark
        print(f"GPU training (depth={args.depth}, devices={args.gpus})...", end=" ", flush=True)
        gpu_time, gpu_info = benchmark_catboost(X_train, y_train, X_test, y_test, "GPU", iterations,
                                                 args.depth, args.gpus)
        print(f"{gpu_time:.2f}s")
        print(f"GPU utilization: {gpu_info}")

        if cpu_time:
            speedup = cpu_time / gpu_time
            print(f"\n>>> SPEEDUP: {speedup:.1f}x <<<")

        results.append({
            "iterations": iterations,
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "speedup": cpu_time / gpu_time if cpu_time else None,
            "gpu_info": gpu_info,
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Iterations':<12} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<12}")
    print("-" * 48)
    for r in results:
        cpu_str = f"{r['cpu_time']:.2f}" if r['cpu_time'] else "N/A"
        speedup_str = f"{r['speedup']:.1f}x" if r['speedup'] else "N/A"
        print(f"{r['iterations']:<12} {cpu_str:<12} {r['gpu_time']:<12.2f} {speedup_str:<12}")

    print("\n" + "=" * 80)
    print("Article claimed: 7.6x (100 iter), 9.8x (500 iter), 9.7x (1000 iter)")
    print("=" * 80)


if __name__ == "__main__":
    main()
