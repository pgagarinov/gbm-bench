#!/usr/bin/env python3
"""
LightGBM Distributed Benchmark: Local vs Remote Dask Cluster

This benchmark compares:
1. Single machine (local, all cores)
2. Remote Dask cluster accessed via SSH tunnel

Note: True 2-machine distributed training isn't possible due to macOS
lo0 routing issues that prevent worker-to-worker communication.
"""

import subprocess
import time
import signal
import sys
import numpy as np

TIMEOUT = 600
REMOTE_HOST = "hvp-dev-mac2"
REMOTE_PYTHON = "/private/tmp/gbm-bench/.pixi/envs/default/bin/python"

DATASET_CONFIGS = [
    {"n_samples": 500_000, "n_features": 50, "name": "500K"},
    {"n_samples": 1_000_000, "n_features": 50, "name": "1M"},
    {"n_samples": 2_000_000, "n_features": 50, "name": "2M"},
    {"n_samples": 5_000_000, "n_features": 50, "name": "5M"},
    {"n_samples": 10_000_000, "n_features": 50, "name": "10M"},
]

def timeout_handler(signum, frame):
    raise TimeoutError("Benchmark timed out")

def cleanup():
    """Clean up all dask processes."""
    subprocess.run(['ssh', REMOTE_HOST, 'pkill -f dask; pkill -f distributed'],
                   capture_output=True, timeout=10)
    subprocess.run(['pkill', '-f', 'dask'], capture_output=True)
    subprocess.run(['pkill', '-f', 'ssh.*-L.*8786'], capture_output=True)
    time.sleep(2)

def setup_remote_cluster(n_workers=4, threads_per_worker=2):
    """Setup Dask cluster on remote via SSH tunnel."""
    processes = []

    # Start scheduler
    sched_cmd = f"{REMOTE_PYTHON} -m distributed.cli.dask_scheduler --host 127.0.0.1 --port 8786"
    sched_proc = subprocess.Popen(
        ['ssh', REMOTE_HOST, sched_cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    processes.append(sched_proc)
    time.sleep(3)

    if sched_proc.poll() is not None:
        raise RuntimeError("Scheduler failed to start")

    # Start workers
    worker_cmd = f"{REMOTE_PYTHON} -m distributed.cli.dask_worker tcp://127.0.0.1:8786 --nworkers {n_workers} --nthreads {threads_per_worker} --memory-limit 4GB --name remote"
    worker_proc = subprocess.Popen(
        ['ssh', REMOTE_HOST, worker_cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    processes.append(worker_proc)
    time.sleep(4)

    if worker_proc.poll() is not None:
        raise RuntimeError("Workers failed to start")

    # Create SSH tunnel
    tunnel_proc = subprocess.Popen(
        ['ssh', '-N', '-L', '8786:127.0.0.1:8786', REMOTE_HOST],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    processes.append(tunnel_proc)
    time.sleep(2)

    if tunnel_proc.poll() is not None:
        raise RuntimeError("SSH tunnel failed")

    return processes

def benchmark_local(X, y, n_estimators=100):
    """Single machine LightGBM."""
    import lightgbm as lgb
    start = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, num_leaves=31, max_depth=6,
        learning_rate=0.1, n_jobs=-1, verbose=-1
    )
    model.fit(X, y)
    return time.time() - start

def benchmark_dask(client, n_samples, n_features, n_estimators=100):
    """Dask LightGBM (data generated on workers)."""
    import dask.array as da
    from lightgbm import DaskLGBMClassifier
    from distributed import wait

    n_workers = len(client.scheduler_info()['workers'])
    chunk_size = max(n_samples // (n_workers * 4), 10000)

    X_dask = da.random.random((n_samples, n_features), chunks=(chunk_size, -1)).astype(np.float32)
    y_dask = da.random.randint(0, 2, n_samples, chunks=chunk_size).astype(np.int32)

    X_dask, y_dask = client.persist([X_dask, y_dask])
    wait([X_dask, y_dask])

    start = time.time()
    model = DaskLGBMClassifier(
        n_estimators=n_estimators, num_leaves=31, max_depth=6,
        learning_rate=0.1, verbose=-1
    )
    model.fit(X_dask, y_dask)
    return time.time() - start

def main():
    print("="*70)
    print("LightGBM Benchmark: Local Machine vs Remote Dask Cluster")
    print("="*70)
    print()

    signal.signal(signal.SIGALRM, timeout_handler)
    processes = None
    results = []

    try:
        cleanup()

        print("[SETUP] Creating Dask cluster on remote machine...")
        processes = setup_remote_cluster(n_workers=4, threads_per_worker=2)

        from distributed import Client
        client = Client('tcp://127.0.0.1:8786', timeout=10)

        info = client.scheduler_info()
        n_workers = len(info['workers'])
        total_threads = sum(w['nthreads'] for w in info['workers'].values())
        print(f"[SETUP] Connected! {n_workers} workers, {total_threads} total threads")
        print()

        for config in DATASET_CONFIGS:
            n_samples = config["n_samples"]
            n_features = config["n_features"]
            name = config["name"]

            print(f"\n{'='*70}")
            print(f"Dataset: {name} ({n_samples:,} x {n_features})")
            print(f"{'='*70}")

            # Generate local dataset
            print("Generating dataset locally...")
            signal.alarm(120)
            np.random.seed(42)
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(np.int32)
            signal.alarm(0)
            print(f"  Size: {X.nbytes / 1e6:.1f} MB")

            # Local benchmark
            print("\n[1] Single Machine (local, all cores)...")
            signal.alarm(TIMEOUT)
            t_local = benchmark_local(X, y)
            signal.alarm(0)
            print(f"    Time: {t_local:.2f}s")

            # Remote Dask benchmark
            print(f"\n[2] Remote Dask Cluster ({n_workers} workers)...")
            signal.alarm(TIMEOUT)
            t_dask = benchmark_dask(client, n_samples, n_features)
            signal.alarm(0)
            print(f"    Time: {t_dask:.2f}s")

            speedup = t_local / t_dask
            print(f"\n[RESULT] Speedup: {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")

            results.append({
                "name": name, "n_samples": n_samples,
                "t_local": t_local, "t_dask": t_dask, "speedup": speedup
            })

            # Free memory
            del X, y

        client.close()

        # Summary
        print("\n" + "="*70)
        print("SUMMARY: LightGBM Training Time Comparison")
        print("="*70)
        print(f"{'Dataset':<10} {'Local(s)':<12} {'Remote Dask':<12} {'Speedup':<10}")
        print("-"*50)
        for r in results:
            marker = "✓" if r['speedup'] > 1 else "✗"
            print(f"{r['name']:<10} {r['t_local']:<12.2f} {r['t_dask']:<12.2f} {r['speedup']:<10.2f}x {marker}")

        # Analysis
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)
        crossover_found = False
        for r in results:
            if r['speedup'] > 1:
                print(f"Dask becomes faster at {r['name']} samples")
                crossover_found = True
                break
        if not crossover_found:
            print("Single machine is faster for all tested dataset sizes.")
            print("This is expected for CPU-only workloads where:")
            print("  - Network overhead exceeds computation savings")
            print("  - LightGBM is already highly optimized for single-node")

    except TimeoutError:
        print(f"\n[ERROR] Timeout after {TIMEOUT}s")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        print("\nCleaning up...")
        if processes:
            for proc in processes:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except:
                    proc.kill()
        cleanup()

    return 0

if __name__ == "__main__":
    sys.exit(main())
