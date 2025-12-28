#!/usr/bin/env python3
"""Approach 1: SSH tunnel for Dask connectivity."""

import subprocess
import time
import signal
import sys

TIMEOUT = 30
REMOTE_HOST = "hvp-dev-mac2"
REMOTE_PYTHON = "/private/tmp/gbm-bench/.pixi/envs/default/bin/python"

def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out")

def main():
    print("="*60)
    print("APPROACH 1: SSH Tunnel")
    print("="*60)
    print("Plan:")
    print("  1. Start scheduler on remote (localhost:8786)")
    print("  2. Start worker on remote (connects to localhost:8786)")
    print("  3. Create SSH tunnel: local:8786 -> remote:8786")
    print("  4. Connect client from local via tunnel")
    print()

    signal.signal(signal.SIGALRM, timeout_handler)

    processes = []

    try:
        # Step 1: Start scheduler on remote
        print("[1/4] Starting scheduler on remote...")
        sched_cmd = f"{REMOTE_PYTHON} -m distributed.cli.dask_scheduler --host 127.0.0.1 --port 8786"
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

        # Step 2: Start worker on remote
        print("[2/4] Starting worker on remote...")
        worker_cmd = f"{REMOTE_PYTHON} -m distributed.cli.dask_worker tcp://127.0.0.1:8786 --nworkers 1 --nthreads 4"
        worker_proc = subprocess.Popen(
            ['ssh', REMOTE_HOST, worker_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append(worker_proc)
        time.sleep(3)

        if worker_proc.poll() is not None:
            _, stderr = worker_proc.communicate()
            print(f"  FAIL: Worker exited: {stderr.decode()}")
            return False
        print("  OK: Worker started")

        # Step 3: Create SSH tunnel
        print("[3/4] Creating SSH tunnel (local:8786 -> remote:8786)...")
        tunnel_proc = subprocess.Popen(
            ['ssh', '-N', '-L', '8786:127.0.0.1:8786', REMOTE_HOST],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append(tunnel_proc)
        time.sleep(2)

        if tunnel_proc.poll() is not None:
            _, stderr = tunnel_proc.communicate()
            print(f"  FAIL: Tunnel exited: {stderr.decode()}")
            return False
        print("  OK: Tunnel created")

        # Step 4: Connect client through tunnel
        print("[4/4] Connecting client via tunnel...")
        signal.alarm(TIMEOUT)

        from distributed import Client
        client = Client('tcp://127.0.0.1:8786', timeout=10)

        info = client.scheduler_info()
        n_workers = len(info['workers'])
        print(f"  OK: Connected! {n_workers} workers")

        # Test computation
        print("\n[TEST] Running computation...")
        import dask.array as da
        x = da.random.random((10000, 10000), chunks=(1000, 1000))
        result = x.sum().compute()
        print(f"  OK: Computed sum = {result:.2f}")

        client.close()
        signal.alarm(0)

        print("\n" + "="*60)
        print("APPROACH 1: SUCCESS")
        print("="*60)
        return True

    except TimeoutError:
        print(f"\n  TIMEOUT after {TIMEOUT}s")
        return False
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False
    finally:
        print("\nCleaning up...")
        for proc in processes:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except:
                proc.kill()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
