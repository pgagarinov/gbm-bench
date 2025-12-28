#!/usr/bin/env python3
"""XGBoost + Dask distributed training test."""

import time
import signal
import sys
import subprocess
import numpy as np

TIMEOUT = 60
REMOTE_HOST = "hvp-dev-mac2"
REMOTE_PYTHON = "/private/tmp/gbm-bench/.pixi/envs/default/bin/python"

def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out")

def cleanup():
    """Clean up Dask processes."""
    subprocess.run(['ssh', REMOTE_HOST, 'pkill -f dask; pkill -f distributed'],
                   capture_output=True, timeout=10)
    subprocess.run(['pkill', '-f', 'dask'], capture_output=True)
    subprocess.run(['pkill', '-f', 'ssh.*-L.*8786'], capture_output=True)
    time.sleep(2)

def test_xgboost_single():
    """Baseline: XGBoost single machine."""
    import xgboost as xgb
    from sklearn.datasets import make_classification

    print("="*60)
    print("TEST 1: XGBoost Single Machine (baseline)")
    print("="*60)

    n_samples = 100_000
    X, y = make_classification(n_samples=n_samples, n_features=50,
                               n_informative=25, random_state=42)
    X = X.astype(np.float32)

    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.1,
        "tree_method": "hist",
        "nthread": -1,
    }

    print("\n[1] Training XGBoost (100 rounds)...")
    start = time.time()
    model = xgb.train(params, dtrain, num_boost_round=100)
    elapsed = time.time() - start
    print(f"    Time: {elapsed:.2f}s")
    return elapsed

def test_xgboost_dask_local():
    """Test XGBoost with local Dask cluster."""
    from dask.distributed import Client, LocalCluster
    import xgboost as xgb
    import xgboost.dask as dxgb
    import dask.array as da

    print("\n" + "="*60)
    print("TEST 2: XGBoost + Dask Local Cluster")
    print("="*60)

    print("\n[1] Creating local Dask cluster...")
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)
    print(f"    Workers: {len(client.scheduler_info()['workers'])}")

    print("\n[2] Generating dataset on workers...")
    n_samples = 100_000
    n_features = 50

    # Generate data on workers
    X = da.random.random((n_samples, n_features), chunks=(25000, n_features)).astype(np.float32)
    y = da.random.randint(0, 2, n_samples, chunks=25000).astype(np.float32)

    print("\n[3] Creating DaskDMatrix...")
    dtrain = dxgb.DaskDMatrix(client, X, y)

    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.1,
        "tree_method": "hist",
    }

    print("\n[4] Training XGBoost with Dask...")
    start = time.time()
    result = dxgb.train(client, params, dtrain, num_boost_round=100)
    elapsed = time.time() - start

    print(f"    Time: {elapsed:.2f}s")
    print(f"    Booster: {result['booster'].num_boosted_rounds()} rounds")

    client.close()
    cluster.close()
    return elapsed

def test_xgboost_dask_remote():
    """Test XGBoost with remote Dask cluster via SSH tunnel."""
    from dask.distributed import Client
    import xgboost.dask as dxgb
    import dask.array as da

    print("\n" + "="*60)
    print("TEST 3: XGBoost + Dask Remote (SSH Tunnel)")
    print("="*60)
    print(f"Remote: {REMOTE_HOST}")

    processes = []

    try:
        cleanup()

        # Start scheduler on remote
        print("\n[1] Starting Dask scheduler on remote...")
        sched_cmd = f"{REMOTE_PYTHON} -m distributed.cli.dask_scheduler --host 127.0.0.1 --port 8786"
        sched_proc = subprocess.Popen(
            ['ssh', REMOTE_HOST, sched_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append(sched_proc)
        time.sleep(3)
        print("    OK: Scheduler started")

        # Start workers on remote
        print("\n[2] Starting Dask workers on remote...")
        worker_cmd = f"{REMOTE_PYTHON} -m distributed.cli.dask_worker tcp://127.0.0.1:8786 --nworkers 2 --nthreads 2 --memory-limit 4GB"
        worker_proc = subprocess.Popen(
            ['ssh', REMOTE_HOST, worker_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append(worker_proc)
        time.sleep(4)
        print("    OK: Workers started")

        # Create SSH tunnel
        print("\n[3] Creating SSH tunnel...")
        tunnel_proc = subprocess.Popen(
            ['ssh', '-N', '-L', '8786:127.0.0.1:8786', REMOTE_HOST],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append(tunnel_proc)
        time.sleep(2)
        print("    OK: Tunnel created")

        # Connect client
        print("\n[4] Connecting client...")
        client = Client('tcp://127.0.0.1:8786', timeout=10)
        n_workers = len(client.scheduler_info()['workers'])
        print(f"    Connected: {n_workers} workers")

        # Generate data on workers
        print("\n[5] Generating dataset on remote workers...")
        n_samples = 100_000
        n_features = 50

        X = da.random.random((n_samples, n_features), chunks=(25000, n_features)).astype(np.float32)
        y = da.random.randint(0, 2, n_samples, chunks=25000).astype(np.float32)

        # Persist to workers
        from distributed import wait
        X, y = client.persist([X, y])
        wait([X, y])
        print("    Data persisted to workers")

        # Create DaskDMatrix
        print("\n[6] Creating DaskDMatrix...")
        dtrain = dxgb.DaskDMatrix(client, X, y)

        params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.1,
            "tree_method": "hist",
        }

        # Train
        print("\n[7] Training XGBoost with Dask...")
        start = time.time()
        result = dxgb.train(client, params, dtrain, num_boost_round=100)
        elapsed = time.time() - start

        print(f"    Time: {elapsed:.2f}s")
        print(f"    Booster: {result['booster'].num_boosted_rounds()} rounds")

        client.close()
        print("\n[RESULT] Remote XGBoost + Dask: SUCCESS")
        return elapsed

    except Exception as e:
        print(f"\n    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        print("\n[8] Cleaning up...")
        for proc in processes:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except:
                proc.kill()
        cleanup()

def main():
    signal.signal(signal.SIGALRM, timeout_handler)

    results = {}

    try:
        # Test 1: Single machine baseline
        signal.alarm(TIMEOUT)
        t_single = test_xgboost_single()
        signal.alarm(0)
        results['single'] = t_single

        # Test 2: Local Dask cluster
        signal.alarm(TIMEOUT)
        t_local = test_xgboost_dask_local()
        signal.alarm(0)
        results['dask_local'] = t_local

        # Test 3: Remote Dask cluster
        signal.alarm(TIMEOUT * 2)  # More time for remote
        t_remote = test_xgboost_dask_remote()
        signal.alarm(0)
        results['dask_remote'] = t_remote

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"XGBoost single machine: {results['single']:.2f}s")
        print(f"XGBoost + Dask local:   {results['dask_local']:.2f}s")
        if results.get('dask_remote'):
            print(f"XGBoost + Dask remote:  {results['dask_remote']:.2f}s")

    except TimeoutError:
        print(f"\n[ERROR] Test timed out after {TIMEOUT}s")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
