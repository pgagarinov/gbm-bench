#!/usr/bin/env python3
"""Approach 2: Scheduler on remote, workers connect via external IPs."""

import subprocess
import time
import signal
import sys
import os

TIMEOUT = 30
LOCAL_IP = "172.16.0.56"
REMOTE_HOST = "hvp-dev-mac2"
REMOTE_IP = "172.16.0.3"
LOCAL_PYTHON = "/Users/peter.gagarinov/_Git/gbm-bench/.pixi/envs/default/bin/python"
REMOTE_PYTHON = "/private/tmp/gbm-bench/.pixi/envs/default/bin/python"

def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out")

def main():
    print("="*60)
    print("APPROACH 2: Scheduler on Remote, External IP Connections")
    print("="*60)
    print("Plan:")
    print(f"  1. Start scheduler on remote ({REMOTE_IP}:8786)")
    print(f"  2. Start worker on remote (connects via localhost)")
    print(f"  3. Start worker on local (connects via {REMOTE_IP}:8786)")
    print(f"  4. Connect client from local via {REMOTE_IP}:8786")
    print()

    signal.signal(signal.SIGALRM, timeout_handler)

    processes = []
    local_worker = None

    try:
        # Step 1: Start scheduler on remote bound to 0.0.0.0
        print("[1/4] Starting scheduler on remote (0.0.0.0:8786)...")
        sched_cmd = f"{REMOTE_PYTHON} -m distributed.cli.dask_scheduler --host 0.0.0.0 --port 8786"
        sched_proc = subprocess.Popen(
            ['ssh', REMOTE_HOST, sched_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append(sched_proc)
        time.sleep(3)

        if sched_proc.poll() is not None:
            _, stderr = sched_proc.communicate()
            print(f"  FAIL: Scheduler exited: {stderr.decode()}")
            return False
        print("  OK: Scheduler started")

        # Step 2: Start worker on remote connecting via localhost
        print("[2/4] Starting worker on remote (via localhost)...")
        worker_cmd = f"{REMOTE_PYTHON} -m distributed.cli.dask_worker tcp://127.0.0.1:8786 --nworkers 1 --nthreads 4 --name remote-worker"
        remote_worker_proc = subprocess.Popen(
            ['ssh', REMOTE_HOST, worker_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append(remote_worker_proc)
        time.sleep(3)

        if remote_worker_proc.poll() is not None:
            _, stderr = remote_worker_proc.communicate()
            print(f"  FAIL: Remote worker exited: {stderr.decode()}")
            return False
        print("  OK: Remote worker started")

        # Step 3: Start worker on local connecting via external IP
        print(f"[3/4] Starting worker on local (connecting to {REMOTE_IP}:8786)...")
        local_worker = subprocess.Popen(
            [LOCAL_PYTHON, '-m', 'distributed.cli.dask_worker',
             f'tcp://{REMOTE_IP}:8786', '--nworkers', '1', '--nthreads', '4',
             '--name', 'local-worker'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
        time.sleep(3)

        if local_worker.poll() is not None:
            _, stderr = local_worker.communicate()
            print(f"  FAIL: Local worker exited: {stderr.decode()}")
            return False
        print("  OK: Local worker started")

        # Step 4: Connect client
        print(f"[4/4] Connecting client to {REMOTE_IP}:8786...")
        signal.alarm(TIMEOUT)

        from distributed import Client
        client = Client(f'tcp://{REMOTE_IP}:8786', timeout=10)

        info = client.scheduler_info()
        n_workers = len(info['workers'])
        worker_names = [w.get('name', 'unknown') for w in info['workers'].values()]
        print(f"  OK: Connected! {n_workers} workers: {worker_names}")

        # Test computation
        print("\n[TEST] Running computation...")
        import dask.array as da
        x = da.random.random((10000, 10000), chunks=(1000, 1000))
        result = x.sum().compute()
        print(f"  OK: Computed sum = {result:.2f}")

        client.close()
        signal.alarm(0)

        print("\n" + "="*60)
        print("APPROACH 2: SUCCESS")
        print("="*60)
        return True

    except TimeoutError:
        print(f"\n  TIMEOUT after {TIMEOUT}s")
        return False
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("\nCleaning up...")
        if local_worker:
            local_worker.terminate()
            try:
                local_worker.wait(timeout=2)
            except:
                local_worker.kill()
        for proc in processes:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except:
                proc.kill()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
