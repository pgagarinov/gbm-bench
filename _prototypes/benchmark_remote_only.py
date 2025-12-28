#!/usr/bin/env python3
"""LightGBM benchmark: Remote Dask cluster accessed via SSH tunnel."""

import subprocess
import time
import signal
import sys
import numpy as np

TIMEOUT = 600
REMOTE_HOST = "hvp-dev-mac2"
REMOTE_PYTHON = "/private/tmp/gbm-bench/.pixi/envs/default/bin/python"

DATASET_CONFIGS = [
    {"n_samples": 1_000_000, "n_features": 50, "name": "1M"},
    {"n_samples": 2_000_000, "n_features": 50, "name": "2M"},
    {"n_samples": 5_000_000, "n_features": 50, "name": "5M"},
]

def timeout_handler(signum, frame):
    raise TimeoutError("Benchmark timed out")

def setup_remote_cluster():
    """Setup Dask cluster entirely on remote, tunnel for client."""
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
        _, stderr = sched_proc.communicate()
        raise RuntimeError(f"Scheduler failed: {stderr.decode()}")

    # Start 4 workers on remote (simulating 2-machine with 2 workers each)
    print("Starting 4 workers on remote...")
    worker_cmd = f"{REMOTE_PYTHON} -m distributed.cli.dask_worker tcp://127.0.0.1:8786 --nworkers 4 --nthreads 2 --memory-limit 4GB"
    worker_proc = subprocess.Popen(
        ['ssh', REMOTE_HOST, worker_cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    processes.append(worker_proc)
    time.sleep(3)

    if worker_proc.poll() is not None:
        _, stderr = worker_proc.communicate()
        raise RuntimeError(f"Workers failed: {stderr.decode()}")

    # Create SSH tunnel
    print("Creating SSH tunnel...")
    tunnel_proc = subprocess.Popen(
        ['ssh', '-N', '-L', '8786:127.0.0.1:8786', REMOTE_HOST],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    processes.append(tunnel_proc)
    time.sleep(2)

    if tunnel_proc.poll() is not None:
        _, stderr = tunnel_proc.communicate()
        raise RuntimeError(f"Tunnel failed: {stderr.decode()}")

    return processes

def benchmark_single(X, y, n_estimators=100):
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
    chunk_size = n_samples // (n_workers * 2)

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
    print("LightGBM: Single Machine vs Remote Dask Cluster (via SSH tunnel)")
    print("="*70)
    print()

    signal.signal(signal.SIGALRM, timeout_handler)
    processes = None
    results = []

    try:
        print("[SETUP] Creating Dask cluster on remote machine...")
        processes = setup_remote_cluster()

        from distributed import Client
        client = Client('tcp://127.0.0.1:8786', timeout=10)

        info = client.scheduler_info()
        n_workers = len(info['workers'])
        print(f"[SETUP] Connected! {n_workers} workers on remote")
        print()

        for config in DATASET_CONFIGS:
            n_samples = config["n_samples"]
            n_features = config["n_features"]
            name = config["name"]

            print(f"\n{'='*70}")
            print(f"Dataset: {name} ({n_samples:,} x {n_features})")
            print(f"{'='*70}")

            # Generate local dataset for single-machine test
            print("Generating dataset locally...")
            signal.alarm(60)
            np.random.seed(42)
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(np.int32)
            signal.alarm(0)
            print(f"  Size: {X.nbytes / 1e6:.1f} MB")

            # Single machine
            print("\n[1] Single Machine (local, all cores)...")
            signal.alarm(TIMEOUT)
            t_single = benchmark_single(X, y)
            signal.alarm(0)
            print(f"    Time: {t_single:.2f}s")

            # Dask on remote
            print(f"\n[2] Dask Cluster (remote, {n_workers} workers)...")
            signal.alarm(TIMEOUT)
            t_dask = benchmark_dask(client, n_samples, n_features)
            signal.alarm(0)
            print(f"    Time: {t_dask:.2f}s")

            speedup = t_single / t_dask
            print(f"\n[RESULT] Speedup: {speedup:.2f}x")

            results.append({
                "name": name, "n_samples": n_samples,
                "t_single": t_single, "t_dask": t_dask, "speedup": speedup
            })

        client.close()

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"{'Dataset':<12} {'Local(s)':<12} {'Remote(s)':<12} {'Speedup':<10}")
        print("-"*46)
        for r in results:
            print(f"{r['name']:<12} {r['t_single']:<12.2f} {r['t_dask']:<12.2f} {r['speedup']:<10.2f}x")

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
        subprocess.run(['ssh', REMOTE_HOST, 'pkill -f dask; pkill -f distributed'],
                       capture_output=True, timeout=10)

    return 0

if __name__ == "__main__":
    sys.exit(main())
