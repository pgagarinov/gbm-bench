#!/usr/bin/env python3
"""XGBoost + Ray distributed training test."""

import time
import signal
import sys
import subprocess
import numpy as np

TIMEOUT = 120
LOCAL_IP = "172.16.0.56"
REMOTE_HOST = "hvp-dev-mac2"
REMOTE_IP = "172.16.0.3"
REMOTE_PYTHON = "/private/tmp/gbm-bench/.pixi/envs/default/bin/python"

def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out")

def cleanup_ray():
    """Clean up Ray processes on both machines."""
    subprocess.run([sys.executable, '-m', 'ray.scripts.scripts', 'stop', '--force'],
                   capture_output=True, timeout=10)
    subprocess.run(['ssh', REMOTE_HOST, f'{REMOTE_PYTHON} -m ray.scripts.scripts stop --force'],
                   capture_output=True, timeout=10)
    subprocess.run(['pkill', '-f', 'ray'], capture_output=True)
    subprocess.run(['pkill', '-f', 'ssh.*-L.*6379'], capture_output=True)
    time.sleep(2)

def test_ray_local():
    """Test Ray with local XGBoost training."""
    import ray
    import xgboost as xgb
    from sklearn.datasets import make_classification

    print("="*60)
    print("TEST 1: Ray Local + XGBoost")
    print("="*60)

    print("\n[1] Initializing Ray...")
    ray.init(num_cpus=4, ignore_reinit_error=True)
    print(f"    Resources: {ray.cluster_resources()}")

    @ray.remote
    def train_xgb(X, y, params, num_rounds):
        import xgboost as xgb
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(params, dtrain, num_boost_round=num_rounds)
        preds = model.predict(dtrain)
        return {'n_trees': model.num_boosted_rounds()}

    # Generate data
    print("\n[2] Generating dataset (100K samples)...")
    X, y = make_classification(n_samples=100_000, n_features=50,
                               n_informative=25, random_state=42)
    X = X.astype(np.float32)

    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.1,
        "tree_method": "hist",
        "nthread": 2,
    }

    # Train multiple models in parallel
    print("\n[3] Training 4 XGBoost models in parallel via Ray...")
    start = time.time()
    futures = [train_xgb.remote(X, y, params, 50) for _ in range(4)]
    results = ray.get(futures)
    t_ray = time.time() - start
    print(f"    Time: {t_ray:.2f}s")
    print(f"    Trained {len(results)} models")

    ray.shutdown()
    return t_ray

def test_single_machine():
    """Test regular XGBoost for comparison."""
    import xgboost as xgb
    from sklearn.datasets import make_classification

    print("\n" + "="*60)
    print("TEST 2: XGBoost Single Machine")
    print("="*60)

    X, y = make_classification(n_samples=100_000, n_features=50,
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
    t_single = time.time() - start
    print(f"    Time: {t_single:.2f}s")
    return t_single

def test_ray_remote_cluster():
    """Test Ray cluster on remote machine via SSH tunnel."""
    import ray

    print("\n" + "="*60)
    print("TEST 3: Ray Remote Cluster (via SSH Tunnel)")
    print("="*60)
    print(f"Remote: {REMOTE_HOST}")

    processes = []

    try:
        cleanup_ray()

        # Start Ray head on remote (bound to localhost)
        print("\n[1] Starting Ray head on remote...")
        head_cmd = f"{REMOTE_PYTHON} -m ray.scripts.scripts start --head --port=6379 --node-ip-address=127.0.0.1 --include-dashboard=false"
        head_proc = subprocess.Popen(
            ['ssh', REMOTE_HOST, head_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append(head_proc)
        time.sleep(5)
        print("    OK: Ray head started on remote")

        # Create SSH tunnel
        print("\n[2] Creating SSH tunnel (local:6379 -> remote:6379)...")
        tunnel_proc = subprocess.Popen(
            ['ssh', '-N', '-L', '6379:127.0.0.1:6379', REMOTE_HOST],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append(tunnel_proc)
        time.sleep(2)
        print("    OK: Tunnel created")

        # Connect to cluster via tunnel
        print("\n[3] Connecting to Ray cluster...")
        ray.init(address='127.0.0.1:6379', ignore_reinit_error=True)
        resources = ray.cluster_resources()
        print(f"    Resources: {resources}")

        nodes = ray.nodes()
        print(f"    Nodes: {len(nodes)}")

        # Run XGBoost training on remote
        @ray.remote
        def train_on_remote(n_samples, n_features, num_rounds):
            import xgboost as xgb
            import numpy as np
            from sklearn.datasets import make_classification
            import socket

            X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                       n_informative=25, random_state=42)
            X = X.astype(np.float32)
            dtrain = xgb.DMatrix(X, label=y)

            params = {
                "objective": "binary:logistic",
                "max_depth": 6,
                "learning_rate": 0.1,
                "tree_method": "hist",
            }

            model = xgb.train(params, dtrain, num_boost_round=num_rounds)
            return {
                'hostname': socket.gethostname(),
                'n_trees': model.num_boosted_rounds(),
            }

        print("\n[4] Training XGBoost on remote via Ray...")
        start = time.time()
        futures = [train_on_remote.remote(100_000, 50, 100) for _ in range(2)]
        results = ray.get(futures)
        elapsed = time.time() - start

        hosts = set(r['hostname'] for r in results)
        print(f"    Time: {elapsed:.2f}s")
        print(f"    Trained on hosts: {hosts}")

        ray.shutdown()
        print("\n[RESULT] Remote Ray cluster: SUCCESS")
        return elapsed

    except Exception as e:
        print(f"\n    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        print("\n[5] Cleaning up...")
        for proc in processes:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except:
                proc.kill()
        cleanup_ray()

def main():
    signal.signal(signal.SIGALRM, timeout_handler)

    try:
        # Test 1: Ray local
        signal.alarm(TIMEOUT)
        t_ray = test_ray_local()
        signal.alarm(0)

        # Test 2: Single machine
        signal.alarm(TIMEOUT)
        t_single = test_single_machine()
        signal.alarm(0)

        # Summary
        print("\n" + "="*60)
        print("LOCAL COMPARISON (100K samples)")
        print("="*60)
        print(f"XGBoost single (100 rounds):     {t_single:.2f}s")
        print(f"Ray parallel (4x50 rounds):      {t_ray:.2f}s")

        # Test 3: Remote cluster
        signal.alarm(TIMEOUT)
        t_remote = test_ray_remote_cluster()
        signal.alarm(0)

        if t_remote:
            print("\n" + "="*60)
            print("FINAL SUMMARY")
            print("="*60)
            print(f"Local single machine:  {t_single:.2f}s")
            print(f"Local Ray (4 actors):  {t_ray:.2f}s")
            print(f"Remote Ray (2 actors): {t_remote:.2f}s")

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
