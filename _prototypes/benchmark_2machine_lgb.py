#!/usr/bin/env python3
"""LightGBM benchmark on 2-machine Dask cluster via SSH tunnel."""

import subprocess
import time
import signal
import sys
import os
import numpy as np

TIMEOUT = 300  # 5 minutes for benchmark
REMOTE_HOST = "hvp-dev-mac2"
REMOTE_PYTHON = "/private/tmp/gbm-bench/.pixi/envs/default/bin/python"

# Dataset sizes to test
DATASET_CONFIGS = [
    {"n_samples": 100_000, "n_features": 50, "name": "100K"},
    {"n_samples": 500_000, "n_features": 50, "name": "500K"},
    {"n_samples": 1_000_000, "n_features": 50, "name": "1M"},
    {"n_samples": 2_000_000, "n_features": 50, "name": "2M"},
]

def timeout_handler(signum, frame):
    raise TimeoutError("Benchmark timed out")

def setup_cluster():
    """Setup Dask cluster via SSH tunnel."""
    processes = []

    # Kill any existing dask processes
    subprocess.run(['ssh', REMOTE_HOST, 'pkill -f dask; pkill -f distributed'],
                   capture_output=True, timeout=10)
    subprocess.run(['pkill', '-f', 'dask'], capture_output=True)
    time.sleep(2)

    # Start scheduler on remote
    print("Starting scheduler on remote...")
    sched_cmd = f"{REMOTE_PYTHON} -m distributed.cli.dask_scheduler --host 127.0.0.1 --port 8786"
    sched_proc = subprocess.Popen(
        ['ssh', REMOTE_HOST, sched_cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    processes.append(sched_proc)
    time.sleep(3)

    if sched_proc.poll() is not None:
        raise RuntimeError("Scheduler failed to start")

    # Start worker on remote (4 threads)
    print("Starting worker on remote...")
    worker_cmd = f"{REMOTE_PYTHON} -m distributed.cli.dask_worker tcp://127.0.0.1:8786 --nworkers 1 --nthreads 8 --memory-limit 8GB --name remote-worker"
    worker_proc = subprocess.Popen(
        ['ssh', REMOTE_HOST, worker_cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    processes.append(worker_proc)
    time.sleep(3)

    if worker_proc.poll() is not None:
        raise RuntimeError("Remote worker failed to start")

    # Create SSH tunnel
    print("Creating SSH tunnel...")
    tunnel_proc = subprocess.Popen(
        ['ssh', '-N', '-L', '8786:127.0.0.1:8786', REMOTE_HOST],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    processes.append(tunnel_proc)
    time.sleep(2)

    if tunnel_proc.poll() is not None:
        raise RuntimeError("SSH tunnel failed")

    # Start local worker (connecting via tunnel)
    print("Starting local worker...")
    local_python = sys.executable
    local_worker = subprocess.Popen(
        [local_python, '-m', 'distributed.cli.dask_worker',
         'tcp://127.0.0.1:8786', '--nworkers', '1', '--nthreads', '8',
         '--memory-limit', '8GB', '--name', 'local-worker'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, 'PYTHONUNBUFFERED': '1'}
    )
    processes.append(local_worker)
    time.sleep(3)

    if local_worker.poll() is not None:
        raise RuntimeError("Local worker failed to start")

    return processes

def benchmark_single_machine(X, y, n_estimators=100):
    """Benchmark LightGBM on single machine."""
    import lightgbm as lgb

    start = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X, y)
    elapsed = time.time() - start

    return elapsed

def benchmark_dask(client, n_samples, n_features, n_estimators=100):
    """Benchmark LightGBM on Dask cluster (data generated on workers)."""
    import dask.array as da
    from lightgbm import DaskLGBMClassifier

    # Generate data directly on workers (no network transfer)
    n_workers = len(client.scheduler_info()['workers'])
    chunk_size = n_samples // (n_workers * 2)  # 2 chunks per worker

    # Create dask arrays - data generated on workers
    X_dask = da.random.random((n_samples, n_features), chunks=(chunk_size, -1)).astype(np.float32)
    y_dask = da.random.randint(0, 2, n_samples, chunks=chunk_size).astype(np.int32)

    # Persist to workers to ensure data is ready
    X_dask, y_dask = client.persist([X_dask, y_dask])
    from distributed import wait
    wait([X_dask, y_dask])

    start = time.time()
    model = DaskLGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.1,
        verbose=-1
    )
    model.fit(X_dask, y_dask)
    elapsed = time.time() - start

    return elapsed

def main():
    print("="*70)
    print("LightGBM Benchmark: Single Machine vs 2-Machine Dask Cluster")
    print("="*70)
    print()

    signal.signal(signal.SIGALRM, timeout_handler)

    processes = None
    results = []

    try:
        # Setup cluster
        print("[SETUP] Creating 2-machine Dask cluster via SSH tunnel...")
        processes = setup_cluster()

        # Connect client
        from distributed import Client
        client = Client('tcp://127.0.0.1:8786', timeout=10)

        info = client.scheduler_info()
        n_workers = len(info['workers'])
        worker_names = [w.get('name', 'unknown') for w in info['workers'].values()]
        print(f"[SETUP] Connected! {n_workers} workers: {worker_names}")
        print()

        # Run benchmarks
        for config in DATASET_CONFIGS:
            n_samples = config["n_samples"]
            n_features = config["n_features"]
            name = config["name"]

            print(f"\n{'='*70}")
            print(f"Dataset: {name} samples ({n_samples:,} x {n_features})")
            print(f"{'='*70}")

            # Generate dataset
            print("Generating dataset...")
            signal.alarm(60)
            np.random.seed(42)
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(np.int32)
            signal.alarm(0)
            print(f"  Dataset size: {X.nbytes / 1e6:.1f} MB")

            # Single machine benchmark
            print("\n[1] Single Machine (all cores)...")
            signal.alarm(TIMEOUT)
            t_single = benchmark_single_machine(X, y)
            signal.alarm(0)
            print(f"    Time: {t_single:.2f}s")

            # Dask cluster benchmark (data generated on workers)
            print("\n[2] Dask Cluster (2 machines, data on workers)...")
            signal.alarm(TIMEOUT)
            t_dask = benchmark_dask(client, n_samples, n_features)
            signal.alarm(0)
            print(f"    Time: {t_dask:.2f}s")

            # Calculate speedup
            speedup = t_single / t_dask
            print(f"\n[RESULT] Speedup: {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")

            results.append({
                "name": name,
                "n_samples": n_samples,
                "t_single": t_single,
                "t_dask": t_dask,
                "speedup": speedup
            })

        client.close()

        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"{'Dataset':<12} {'Single(s)':<12} {'Dask 2M(s)':<12} {'Speedup':<10}")
        print("-"*46)
        for r in results:
            print(f"{r['name']:<12} {r['t_single']:<12.2f} {r['t_dask']:<12.2f} {r['speedup']:<10.2f}x")

        print("\n" + "="*70)
        print("BENCHMARK COMPLETE")
        print("="*70)

    except TimeoutError:
        print(f"\n[ERROR] Benchmark timed out after {TIMEOUT}s")
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
        # Also cleanup remote
        subprocess.run(['ssh', REMOTE_HOST, 'pkill -f dask; pkill -f distributed'],
                       capture_output=True, timeout=10)

    return 0

if __name__ == "__main__":
    sys.exit(main())
