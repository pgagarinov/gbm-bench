#!/usr/bin/env python3
"""
XGBoost + Dask benchmark on 2-Mac cluster.

Two modes available:
1. REMOTE mode (default): Workers on remote, tracker on local
   - Avoids macOS lo0 routing issue
   - Requires network connectivity from remote to local

2. SSH mode (--ssh-mode): Everything on remote via SSH
   - More reliable if firewall blocks incoming connections
   - Runs training script on remote machine

Usage:
    # Default mode (workers on remote):
    python benchmark_2mac_xgboost.py

    # SSH mode (everything on remote):
    python benchmark_2mac_xgboost.py --ssh-mode

    # Custom IPs:
    python benchmark_2mac_xgboost.py --local-ip 172.16.0.56 --remote-host hvp-dev-mac2
"""

import argparse
import subprocess
import time
import signal
import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score

# Default configuration
LOCAL_IP = "172.16.0.56"
REMOTE_HOST = "hvp-dev-mac2"
REMOTE_IP = "172.16.0.3"
REMOTE_PYTHON = "/private/tmp/gbm-bench/.pixi/envs/default/bin/python"
REMOTE_PIXI = "/opt/homebrew/bin/pixi"
REMOTE_DIR = "/tmp/gbm-bench"

# Note on the macOS lo0 issue:
# macOS routes traffic to its own external IP through the loopback interface (lo0).
# This breaks XGBoost's Rabit tracker when workers and tracker are on the same machine.
#
# Solution: Run all workers on a DIFFERENT machine than the tracker.
# - Default mode: Workers on remote, tracker on local (client machine)
# - SSH mode: Everything on remote, avoiding the issue entirely
#
# For a true 2-machine distributed setup where BOTH machines run workers,
# you would need the tracker on a third machine without workers.

# Benchmark parameters
N_SAMPLES = 1_000_000
N_FEATURES = 50
ITERATIONS = 100
DEPTH = 6
LEARNING_RATE = 0.1
TIMEOUT = 120


def cleanup_remote(remote_host):
    """Clean up Dask processes on remote machine."""
    print("Cleaning up remote processes...")
    subprocess.run(
        ["ssh", remote_host, "pkill -f 'dask-scheduler'; pkill -f 'dask-worker'; pkill -f 'dask worker'"],
        capture_output=True,
        timeout=10
    )
    time.sleep(1)


def cleanup_local():
    """Clean up local SSH processes."""
    subprocess.run(["pkill", "-f", "ssh.*-L.*8786"], capture_output=True)
    subprocess.run(["pkill", "-f", "ssh.*dask"], capture_output=True)


def start_remote_cluster(local_ip, remote_host, n_workers=2, threads_per_worker=4):
    """
    Start Dask scheduler and workers on remote machine.

    Returns:
        tuple: (scheduler_proc, worker_proc, tunnel_proc)
    """
    processes = []

    # Start scheduler on remote - bind to ALL interfaces so local can connect
    print(f"\n[1] Starting Dask scheduler on {remote_host}...")
    # Use 0.0.0.0 to accept connections from any interface
    sched_cmd = f"cd {REMOTE_DIR} && {REMOTE_PIXI} run dask scheduler --host 0.0.0.0 --port 8786"
    sched_proc = subprocess.Popen(
        ["ssh", "-o", "StrictHostKeyChecking=no", remote_host, sched_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    processes.append(sched_proc)
    time.sleep(3)

    if sched_proc.poll() is not None:
        stdout, stderr = sched_proc.communicate()
        print(f"ERROR: Scheduler failed to start")
        print(f"stdout: {stdout.decode()}")
        print(f"stderr: {stderr.decode()}")
        return None
    print("    Scheduler started")

    # Start workers on remote - connect via localhost (same machine)
    print(f"\n[2] Starting {n_workers} workers on {remote_host}...")
    worker_cmd = (
        f"cd {REMOTE_DIR} && {REMOTE_PIXI} run dask worker tcp://127.0.0.1:8786 "
        f"--nworkers {n_workers} --nthreads {threads_per_worker} "
        f"--memory-limit 8GB --no-dashboard"
    )
    worker_proc = subprocess.Popen(
        ["ssh", "-o", "StrictHostKeyChecking=no", remote_host, worker_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    processes.append(worker_proc)
    time.sleep(4)

    if worker_proc.poll() is not None:
        stdout, stderr = worker_proc.communicate()
        print(f"ERROR: Workers failed to start")
        print(f"stderr: {stderr.decode()}")
        return None
    print("    Workers started")

    return processes


def run_baseline_xgboost(X_train, y_train, X_test, y_test):
    """Run single-machine XGBoost as baseline."""
    import xgboost as xgb

    print(f"\n{'='*60}")
    print("BASELINE: XGBoost Single Machine")
    print("="*60)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": DEPTH,
        "learning_rate": LEARNING_RATE,
        "nthread": -1,
    }

    start = time.time()
    model = xgb.train(params, dtrain, num_boost_round=ITERATIONS, verbose_eval=False)
    train_time = time.time() - start

    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"  Time: {train_time:.2f}s")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")

    return {"time": train_time, "accuracy": accuracy, "auc": auc}


def run_dask_xgboost(X_train, y_train, X_test, y_test, client, local_ip):
    """
    Run XGBoost with Dask, using explicit tracker configuration.

    The key fix: Use xgboost.collective.Config to set tracker_host_ip
    to the local machine's IP, so remote workers can connect back.
    """
    import xgboost as xgb
    import xgboost.dask as dxgb
    import dask.array as da

    # Try to import the collective Config (XGBoost 3.x)
    try:
        from xgboost.collective import Config as CollConfig
        has_coll_config = True
    except ImportError:
        has_coll_config = False
        print("  Warning: xgboost.collective.Config not available (XGBoost < 3.0)")

    n_workers = len(client.scheduler_info()['workers'])
    total_threads = sum(w['nthreads'] for w in client.scheduler_info()['workers'].values())

    print(f"\n{'='*60}")
    print(f"XGBoost + Dask ({n_workers} workers, {total_threads} threads)")
    print("="*60)

    # Convert to Dask arrays - chunk by number of workers
    chunk_size = max(1, len(X_train) // (n_workers * 2))
    print(f"  Chunk size: {chunk_size:,} samples")

    X_da = da.from_array(X_train, chunks=(chunk_size, -1))
    y_da = da.from_array(y_train, chunks=(chunk_size,))

    # Persist data to workers
    print("  Distributing data to workers...")
    X_da, y_da = client.persist([X_da, y_da])
    from distributed import wait
    wait([X_da, y_da])
    print("  Data distributed")

    # Create DaskDMatrix
    print("  Creating DaskDMatrix...")
    dtrain = dxgb.DaskDMatrix(client, X_da, y_da)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": DEPTH,
        "learning_rate": LEARNING_RATE,
    }

    # Configure the collective communicator with explicit tracker IP
    # This tells workers to connect back to our machine for coordination
    print(f"  Starting training (tracker on {local_ip})...")
    start = time.time()

    if has_coll_config:
        # XGBoost 3.x: Use collective Config
        # KEY FIX: For remote workers connecting back to local tracker,
        # use local_ip. For local-only clusters, use 127.0.0.1.
        coll_cfg = CollConfig(
            retry=3,
            timeout=60,
            tracker_host_ip=local_ip,
            tracker_port=0,  # Auto-select port
        )
        output = dxgb.train(
            client, params, dtrain,
            num_boost_round=ITERATIONS,
            verbose_eval=False,
            coll_cfg=coll_cfg
        )
    else:
        # Fallback: Try with dask config
        import dask
        with dask.config.set({"xgboost.scheduler_address": local_ip}):
            output = dxgb.train(
                client, params, dtrain,
                num_boost_round=ITERATIONS,
                verbose_eval=False
            )

    train_time = time.time() - start

    model = output['booster']

    # Predict on test set (using regular XGBoost since test is small)
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"  Time: {train_time:.2f}s")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")

    return {"time": train_time, "accuracy": accuracy, "auc": auc}


def run_ssh_mode(args, X_train, y_train, X_test, y_test):
    """
    Run everything on remote machine via SSH.

    This avoids cross-machine Rabit communication by running
    scheduler, workers, AND training all on the remote machine.
    """
    import tempfile
    import pickle

    print(f"\n{'='*60}")
    print("SSH MODE: Running XGBoost + Dask entirely on remote")
    print("="*60)

    # Save data to temp file
    print("\n[1] Saving data to transfer...")
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        data_file = f.name
        pickle.dump({
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'params': {
                'iterations': ITERATIONS,
                'depth': DEPTH,
                'learning_rate': LEARNING_RATE,
                'workers': args.workers,
                'threads': args.threads,
            }
        }, f)

    remote_data = f"/tmp/xgb_data_{int(time.time())}.pkl"

    try:
        # Transfer data
        print(f"\n[2] Transferring data to {args.remote_host}...")
        subprocess.run(
            ["scp", data_file, f"{args.remote_host}:{remote_data}"],
            check=True, capture_output=True
        )
        print("    Data transferred")

        # Create remote training script
        remote_script = f'''
import pickle
import time
import numpy as np
from dask.distributed import Client, LocalCluster
import xgboost as xgb
import xgboost.dask as dxgb
import dask.array as da
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
with open("{remote_data}", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]
params = data["params"]

print(f"Data loaded: {{X_train.shape}}")

# Create local cluster on this machine
print("Creating local Dask cluster...")
cluster = LocalCluster(
    n_workers=params["workers"],
    threads_per_worker=params["threads"],
    memory_limit="8GB",
)
client = Client(cluster)
n_workers = len(client.scheduler_info()["workers"])
print(f"Cluster ready: {{n_workers}} workers")

# Convert to Dask arrays
chunk_size = max(1, len(X_train) // (n_workers * 2))
X_da = da.from_array(X_train, chunks=(chunk_size, -1))
y_da = da.from_array(y_train, chunks=(chunk_size,))

# Create DMatrix
print("Creating DaskDMatrix...")
dtrain = dxgb.DaskDMatrix(client, X_da, y_da)

xgb_params = {{
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": params["depth"],
    "learning_rate": params["learning_rate"],
}}

# Train
print("Training XGBoost with Dask...")
from xgboost.collective import Config as CollConfig
# KEY FIX: Use 127.0.0.1 for tracker on macOS
coll_cfg = CollConfig(retry=3, timeout=60, tracker_host_ip="127.0.0.1", tracker_port=0)
start = time.time()
output = dxgb.train(client, xgb_params, dtrain, num_boost_round=params["iterations"], coll_cfg=coll_cfg)
train_time = time.time() - start

model = output["booster"]

# Predict
dtest = xgb.DMatrix(X_test)
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"RESULT:time={{train_time:.4f}},accuracy={{accuracy:.4f}},auc={{auc:.4f}}")

client.close()
cluster.close()
'''

        # Write script to remote
        print(f"\n[3] Running training on {args.remote_host}...")
        result = subprocess.run(
            ["ssh", args.remote_host,
             f"cd {REMOTE_DIR} && {REMOTE_PIXI} run python -c '{remote_script}'"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if args.debug:
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")

        # Parse result
        for line in result.stdout.split('\n'):
            if line.startswith("RESULT:"):
                parts = line.replace("RESULT:", "").split(",")
                metrics = {}
                for part in parts:
                    key, val = part.split("=")
                    metrics[key] = float(val)

                print(f"\n  Time: {metrics['time']:.2f}s")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  AUC: {metrics['auc']:.4f}")

                return metrics

        print(f"ERROR: Could not parse result")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return None

    finally:
        # Cleanup
        import os
        os.unlink(data_file)
        subprocess.run(
            ["ssh", args.remote_host, f"rm -f {remote_data}"],
            capture_output=True
        )


def main():
    parser = argparse.ArgumentParser(description="XGBoost + Dask on 2-Mac cluster")
    parser.add_argument("--local-ip", default=LOCAL_IP, help="Local machine IP")
    parser.add_argument("--remote-host", default=REMOTE_HOST, help="Remote hostname")
    parser.add_argument("--remote-ip", default=REMOTE_IP, help="Remote machine IP")
    parser.add_argument("--workers", type=int, default=2, help="Workers on remote")
    parser.add_argument("--threads", type=int, default=4, help="Threads per worker")
    parser.add_argument("--samples", type=int, default=N_SAMPLES, help="Number of samples")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline test")
    parser.add_argument("--ssh-mode", action="store_true", help="Run everything on remote via SSH")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    print("="*60)
    print("XGBoost + Dask: 2-Mac Cluster Benchmark")
    print("="*60)
    print(f"Local:  {args.local_ip}")
    print(f"Remote: {args.remote_host} ({args.remote_ip})")
    print(f"Workers: {args.workers} x {args.threads} threads")

    # Generate data
    print(f"\nGenerating {args.samples:,} samples...")
    X, y = make_classification(
        n_samples=args.samples,
        n_features=N_FEATURES,
        n_informative=30,
        n_redundant=10,
        random_state=42
    )
    X = X.astype(np.float32)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results = {}

    # Baseline
    if not args.skip_baseline:
        results['baseline'] = run_baseline_xgboost(X_train, y_train, X_test, y_test)

    # SSH mode: run everything on remote
    if args.ssh_mode:
        dask_result = run_ssh_mode(args, X_train, y_train, X_test, y_test)
        if dask_result:
            results['dask'] = dask_result
    else:
        # Remote mode: workers on remote, client/tracker on local
        cleanup_local()
        cleanup_remote(args.remote_host)

        processes = []
        try:
            # Start remote cluster
            processes = start_remote_cluster(
                args.local_ip, args.remote_host,
                n_workers=args.workers,
                threads_per_worker=args.threads
            )

            if not processes:
                print("\nFailed to start remote cluster")
                return 1

            # Connect from local
            from dask.distributed import Client

            print(f"\n[3] Connecting to remote scheduler at {args.remote_ip}:8786...")
            scheduler_addr = f"tcp://{args.remote_ip}:8786"
            client = Client(scheduler_addr, timeout=30)

            n_workers = len(client.scheduler_info()['workers'])
            print(f"    Connected: {n_workers} workers")

            if n_workers == 0:
                print("ERROR: No workers available")
                return 1

            # Run Dask XGBoost
            results['dask'] = run_dask_xgboost(
                X_train, y_train, X_test, y_test,
                client, args.local_ip
            )

            client.close()

        except TimeoutError:
            print(f"\nERROR: Operation timed out")
            return 1
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            # Cleanup
            print("\nCleaning up...")
            for proc in processes:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except:
                    proc.kill()
            cleanup_remote(args.remote_host)
            cleanup_local()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if 'baseline' in results:
        print(f"\nBaseline (local): {results['baseline']['time']:.2f}s")

    if 'dask' in results:
        mode = "SSH remote" if args.ssh_mode else "Dask remote"
        print(f"{mode}:       {results['dask']['time']:.2f}s")

        if 'baseline' in results:
            speedup = results['baseline']['time'] / results['dask']['time']
            print(f"Speedup:          {speedup:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
