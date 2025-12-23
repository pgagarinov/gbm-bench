#!/usr/bin/env python3
"""
Comprehensive benchmark for XGBoost and CatBoost with:
1. Accuracy/AUC comparison for fair evaluation
2. JSON output with hardware configuration
3. XGBoost multi-GPU via Dask
"""

import argparse
import json
import os
import platform
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import catboost as cb
import xgboost as xgb


class GPUMonitor:
    """Monitor GPU utilization during training."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.running = False
        self.thread = None
        self.snapshots = []

    def _monitor(self):
        while self.running:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    snapshot = {}
                    for line in result.stdout.strip().split("\n"):
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 4:
                            gpu_id = int(parts[0])
                            snapshot[gpu_id] = {
                                "util": int(parts[1]),
                                "mem_used": int(parts[2]),
                                "mem_total": int(parts[3])
                            }
                    self.snapshots.append(snapshot)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.snapshots = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def get_stats(self):
        if not self.snapshots:
            return {}
        all_gpus = set()
        for s in self.snapshots:
            all_gpus.update(s.keys())
        stats = {}
        for gpu_id in sorted(all_gpus):
            utils = [s[gpu_id]["util"] for s in self.snapshots if gpu_id in s]
            if utils:
                stats[f"gpu{gpu_id}_avg_util"] = round(np.mean(utils), 1)
                stats[f"gpu{gpu_id}_max_util"] = max(utils)
        return stats


def get_hardware_info():
    """Collect hardware information."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    # CPU info
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "Model name:" in line:
                info["cpu_model"] = line.split(":")[1].strip()
            elif "CPU(s):" in line and "NUMA" not in line and "On-line" not in line:
                info["cpu_count"] = int(line.split(":")[1].strip())
    except Exception:
        pass

    # GPU info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    mem_str = parts[1].replace(" MiB", "").strip()
                    gpus.append({
                        "name": parts[0],
                        "memory_mb": int(mem_str) if mem_str.isdigit() else 0,
                        "driver": parts[2],
                    })
            info["gpus"] = gpus
            info["gpu_count"] = len(gpus)

        # Get CUDA version separately
        cuda_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if cuda_result.returncode == 0:
            # Also get CUDA version from nvcc if available
            nvcc_result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
            if nvcc_result.returncode == 0:
                for line in nvcc_result.stdout.split("\n"):
                    if "release" in line.lower():
                        info["cuda_version"] = line.strip()
                        break
    except Exception as e:
        info["gpu_error"] = str(e)

    # Library versions
    info["xgboost_version"] = xgb.__version__
    info["catboost_version"] = cb.__version__

    return info


def generate_dataset(n_samples: int, n_features: int, random_state: int = 42):
    """Generate synthetic classification dataset."""
    print(f"Generating dataset: {n_samples:,} samples x {n_features} features...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=n_features // 4,
        n_clusters_per_class=2,
        flip_y=0.01,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=random_state
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def benchmark_xgboost_single(X_train, y_train, X_test, y_test, device: str,
                              iterations: int, depth: int, learning_rate: float):
    """Benchmark XGBoost on single device (CPU or 1 GPU)."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": depth,
        "eta": learning_rate,
        "verbosity": 0,
    }

    if device == "cpu":
        params["device"] = "cpu"
        params["nthread"] = -1
    else:
        params["device"] = "cuda:0"
        params["tree_method"] = "hist"

    monitor = GPUMonitor()
    if device != "cpu":
        monitor.start()

    start = time.perf_counter()
    model = xgb.train(params, dtrain, num_boost_round=iterations,
                      evals=[(dtest, "test")], verbose_eval=False)
    elapsed = time.perf_counter() - start

    if device != "cpu":
        monitor.stop()

    # Predictions and metrics
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    return {
        "time_seconds": round(elapsed, 2),
        "accuracy": round(accuracy, 4),
        "auc": round(auc, 4),
        "gpu_stats": monitor.get_stats() if device != "cpu" else {}
    }


def benchmark_xgboost_dask(X_train, y_train, X_test, y_test,
                           iterations: int, depth: int, learning_rate: float, n_gpus: int):
    """Benchmark XGBoost with Dask for multi-GPU using LocalCUDACluster."""
    import xgboost.dask as xgb_dask
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    from dask import array as da

    print(f"  Setting up LocalCUDACluster with {n_gpus} GPUs...")

    # LocalCUDACluster assigns each worker to a separate GPU
    cluster = LocalCUDACluster(
        n_workers=n_gpus,
        CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(n_gpus)),
    )
    client = Client(cluster)

    try:
        # Convert to Dask arrays
        chunk_size = len(X_train) // n_gpus
        X_train_da = da.from_array(X_train, chunks=(chunk_size, -1))
        y_train_da = da.from_array(y_train, chunks=(chunk_size,))

        dtrain = xgb_dask.DaskDMatrix(client, X_train_da, y_train_da)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": depth,
            "eta": learning_rate,
            "tree_method": "hist",
            "device": "cuda",
            "verbosity": 0,
        }

        monitor = GPUMonitor()
        monitor.start()

        start = time.perf_counter()
        output = xgb_dask.train(
            client, params, dtrain,
            num_boost_round=iterations,
            evals=[(dtrain, "train")]
        )
        elapsed = time.perf_counter() - start

        monitor.stop()

        model = output["booster"]

        # Predictions on CPU
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        return {
            "time_seconds": round(elapsed, 2),
            "accuracy": round(accuracy, 4),
            "auc": round(auc, 4),
            "gpu_stats": monitor.get_stats()
        }
    finally:
        client.close()
        cluster.close()


def benchmark_catboost(X_train, y_train, X_test, y_test, device: str,
                       iterations: int, depth: int, learning_rate: float, devices: str = "0"):
    """Benchmark CatBoost."""
    params = {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
        "loss_function": "Logloss",
        "verbose": False,
    }

    if device == "cpu":
        params["task_type"] = "CPU"
        params["thread_count"] = -1
    else:
        params["task_type"] = "GPU"
        params["devices"] = devices

    model = cb.CatBoostClassifier(**params)

    monitor = GPUMonitor()
    if device != "cpu":
        monitor.start()

    start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    elapsed = time.perf_counter() - start

    if device != "cpu":
        monitor.stop()

    # Predictions and metrics
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    return {
        "time_seconds": round(elapsed, 2),
        "accuracy": round(accuracy, 4),
        "auc": round(auc, 4),
        "gpu_stats": monitor.get_stats() if device != "cpu" else {}
    }


def run_benchmarks(args):
    """Run all benchmarks."""
    results = {
        "hardware": get_hardware_info(),
        "parameters": {
            "samples": args.samples,
            "features": args.features,
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
        },
        "benchmarks": []
    }

    # Generate dataset once
    X_train, X_test, y_train, y_test = generate_dataset(args.samples, args.features)

    for iterations in args.iterations:
        print(f"\n{'='*70}")
        print(f"ITERATIONS: {iterations}")
        print("=" * 70)

        bench_results = {"iterations": iterations, "results": {}}

        # XGBoost CPU
        if not args.skip_cpu:
            print("\nXGBoost CPU...", end=" ", flush=True)
            try:
                res = benchmark_xgboost_single(
                    X_train, y_train, X_test, y_test, "cpu",
                    iterations, args.depth, args.learning_rate
                )
                print(f"{res['time_seconds']}s | Acc: {res['accuracy']} | AUC: {res['auc']}")
                bench_results["results"]["xgboost_cpu"] = res
            except Exception as e:
                print(f"Error: {e}")

        # XGBoost 1 GPU
        if not args.skip_1gpu:
            print("XGBoost 1 GPU...", end=" ", flush=True)
            try:
                res = benchmark_xgboost_single(
                    X_train, y_train, X_test, y_test, "gpu",
                    iterations, args.depth, args.learning_rate
                )
                print(f"{res['time_seconds']}s | Acc: {res['accuracy']} | AUC: {res['auc']}")
                bench_results["results"]["xgboost_1gpu"] = res
            except Exception as e:
                print(f"Error: {e}")

        # XGBoost multi-GPU with Dask
        if not args.skip_multi_gpu and args.gpus > 1:
            print(f"XGBoost {args.gpus} GPUs (Dask)...", end=" ", flush=True)
            try:
                res = benchmark_xgboost_dask(
                    X_train, y_train, X_test, y_test,
                    iterations, args.depth, args.learning_rate, args.gpus
                )
                print(f"{res['time_seconds']}s | Acc: {res['accuracy']} | AUC: {res['auc']}")
                bench_results["results"][f"xgboost_{args.gpus}gpu_dask"] = res
            except Exception as e:
                print(f"Error: {e}")

        # CatBoost CPU
        if not args.skip_cpu:
            print("CatBoost CPU...", end=" ", flush=True)
            try:
                res = benchmark_catboost(
                    X_train, y_train, X_test, y_test, "cpu",
                    iterations, args.depth, args.learning_rate
                )
                print(f"{res['time_seconds']}s | Acc: {res['accuracy']} | AUC: {res['auc']}")
                bench_results["results"]["catboost_cpu"] = res
            except Exception as e:
                print(f"Error: {e}")

        # CatBoost 1 GPU
        if not args.skip_1gpu:
            print("CatBoost 1 GPU...", end=" ", flush=True)
            try:
                res = benchmark_catboost(
                    X_train, y_train, X_test, y_test, "gpu",
                    iterations, args.depth, args.learning_rate, "0"
                )
                print(f"{res['time_seconds']}s | Acc: {res['accuracy']} | AUC: {res['auc']}")
                bench_results["results"]["catboost_1gpu"] = res
            except Exception as e:
                print(f"Error: {e}")

        # CatBoost multi-GPU
        if not args.skip_multi_gpu and args.gpus > 1:
            print(f"CatBoost {args.gpus} GPUs...", end=" ", flush=True)
            try:
                devices = f"0-{args.gpus - 1}"
                res = benchmark_catboost(
                    X_train, y_train, X_test, y_test, "gpu",
                    iterations, args.depth, args.learning_rate, devices
                )
                print(f"{res['time_seconds']}s | Acc: {res['accuracy']} | AUC: {res['auc']}")
                bench_results["results"][f"catboost_{args.gpus}gpu"] = res
            except Exception as e:
                print(f"Error: {e}")

        results["benchmarks"].append(bench_results)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for bench in results["benchmarks"]:
        print(f"\nIterations: {bench['iterations']}")
        print(f"{'Config':<25} {'Time (s)':<12} {'Accuracy':<12} {'AUC':<12} {'Speedup':<12}")
        print("-" * 70)

        cpu_times = {}
        for name, res in bench["results"].items():
            if "cpu" in name:
                lib = name.split("_")[0]
                cpu_times[lib] = res["time_seconds"]

        for name, res in bench["results"].items():
            lib = name.split("_")[0]
            cpu_time = cpu_times.get(lib)
            speedup = f"{cpu_time / res['time_seconds']:.1f}x" if cpu_time and "cpu" not in name else "-"
            print(f"{name:<25} {res['time_seconds']:<12} {res['accuracy']:<12} {res['auc']:<12} {speedup:<12}")

    # Save to JSON
    output_file = args.output or f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def run_single_test(test_name: str, args):
    """Run a single quick test."""
    print("=" * 70)
    print(f"Quick Test: {test_name}")
    print("=" * 70)
    print(f"Dataset: {args.samples:,} samples x {args.features} features")
    print(f"Iterations: {args.iterations[0]}, Depth: {args.depth}")
    print("=" * 70)

    X_train, X_test, y_train, y_test = generate_dataset(args.samples, args.features)

    iterations = args.iterations[0]

    if test_name == "xgb-4gpu":
        print(f"\nXGBoost {args.gpus} GPUs (Dask)...")
        res = benchmark_xgboost_dask(
            X_train, y_train, X_test, y_test,
            iterations, args.depth, args.learning_rate, args.gpus
        )
    elif test_name == "xgb-1gpu":
        print("\nXGBoost 1 GPU...")
        res = benchmark_xgboost_single(
            X_train, y_train, X_test, y_test, "gpu",
            iterations, args.depth, args.learning_rate
        )
    elif test_name == "xgb-cpu":
        print("\nXGBoost CPU...")
        res = benchmark_xgboost_single(
            X_train, y_train, X_test, y_test, "cpu",
            iterations, args.depth, args.learning_rate
        )
    elif test_name == "cb-4gpu":
        print(f"\nCatBoost {args.gpus} GPUs...")
        devices = f"0-{args.gpus - 1}"
        res = benchmark_catboost(
            X_train, y_train, X_test, y_test, "gpu",
            iterations, args.depth, args.learning_rate, devices
        )
    elif test_name == "cb-1gpu":
        print("\nCatBoost 1 GPU...")
        res = benchmark_catboost(
            X_train, y_train, X_test, y_test, "gpu",
            iterations, args.depth, args.learning_rate, "0"
        )
    elif test_name == "cb-cpu":
        print("\nCatBoost CPU...")
        res = benchmark_catboost(
            X_train, y_train, X_test, y_test, "cpu",
            iterations, args.depth, args.learning_rate
        )
    else:
        raise ValueError(f"Unknown test: {test_name}")

    print(f"\nResults:")
    print(f"  Time: {res['time_seconds']}s")
    print(f"  Accuracy: {res['accuracy']}")
    print(f"  AUC: {res['auc']}")
    if res.get('gpu_stats'):
        print(f"  GPU Stats: {res['gpu_stats']}")

    return res


def main():
    parser = argparse.ArgumentParser(description="Full XGBoost/CatBoost benchmark")

    # Quick test mode
    parser.add_argument("--test", choices=["xgb-cpu", "xgb-1gpu", "xgb-4gpu", "cb-cpu", "cb-1gpu", "cb-4gpu"],
                        help="Run a single quick test")

    parser.add_argument("--samples", type=int, default=10_500_000)
    parser.add_argument("--features", type=int, default=28)
    parser.add_argument("--iterations", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-1gpu", action="store_true")
    parser.add_argument("--skip-multi-gpu", action="store_true")
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--skip-catboost", action="store_true")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    if args.test:
        run_single_test(args.test, args)
        return

    print("=" * 70)
    print("XGBoost & CatBoost Comprehensive Benchmark")
    print("=" * 70)
    print(f"Dataset: {args.samples:,} samples x {args.features} features")
    print(f"Depth: {args.depth}, Learning rate: {args.learning_rate}")
    print(f"Iterations: {args.iterations}")
    print(f"GPUs for multi-GPU tests: {args.gpus}")
    print("=" * 70)

    run_benchmarks(args)


if __name__ == "__main__":
    main()
