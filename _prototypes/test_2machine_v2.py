#!/usr/bin/env python3
"""2-machine Dask test with explicit contact addresses."""

import subprocess
import time
import signal
import sys
import os

TIMEOUT = 60
LOCAL_IP = "172.16.0.56"
REMOTE_HOST = "hvp-dev-mac2"
REMOTE_IP = "172.16.0.3"
LOCAL_PYTHON = sys.executable
REMOTE_PYTHON = "/private/tmp/gbm-bench/.pixi/envs/default/bin/python"

def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out")

def main():
    print("="*70)
    print("2-Machine Dask Test with Explicit Contact Addresses")
    print("="*70)
    print(f"Local: {LOCAL_IP}")
    print(f"Remote: {REMOTE_HOST} ({REMOTE_IP})")
    print()
    print("Network routes:")
    print(f"  Local → {REMOTE_IP}: WORKS (different machine)")
    print(f"  Remote → {LOCAL_IP}: WORKS (different machine)")
    print(f"  Local → {LOCAL_IP}: BROKEN (lo0 routing)")
    print(f"  Remote → {REMOTE_IP}: BROKEN (lo0 routing)")
    print()

    signal.signal(signal.SIGALRM, timeout_handler)
    processes = []
    local_worker = None

    try:
        # Cleanup
        subprocess.run(['ssh', REMOTE_HOST, 'pkill -f dask; pkill -f distributed'],
                       capture_output=True, timeout=10)
        subprocess.run(['pkill', '-f', 'dask'], capture_output=True)
        time.sleep(2)

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
            print(f"  FAIL: {stderr.decode()[:200]}")
            return False
        print("  OK: Scheduler started")

        # Step 2: Start worker on remote (connects via localhost, advertises external IP)
        print("[2/4] Starting worker on remote...")
        # Remote worker: connects to scheduler via localhost (works)
        # Advertises itself at external IP (so local worker can reach it)
        worker_cmd = f"{REMOTE_PYTHON} -m distributed.cli.dask_worker tcp://127.0.0.1:8786 --nworkers 1 --nthreads 4 --name remote-worker --listen-address tcp://0.0.0.0:9001 --contact-address tcp://{REMOTE_IP}:9001"
        remote_worker = subprocess.Popen(
            ['ssh', REMOTE_HOST, worker_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append(remote_worker)
        time.sleep(4)

        if remote_worker.poll() is not None:
            _, stderr = remote_worker.communicate()
            print(f"  FAIL: {stderr.decode()[:200]}")
            return False
        print("  OK: Remote worker started")

        # Step 3: Start worker on local (connects via remote IP, advertises local IP)
        print("[3/4] Starting worker on local...")
        # Local worker: connects to scheduler via remote IP (works)
        # Advertises itself at local IP (so remote worker can reach it)
        local_worker = subprocess.Popen(
            [LOCAL_PYTHON, '-m', 'distributed.cli.dask_worker',
             f'tcp://{REMOTE_IP}:8786',
             '--nworkers', '1', '--nthreads', '4',
             '--name', 'local-worker',
             '--listen-address', 'tcp://0.0.0.0:9002',
             '--contact-address', f'tcp://{LOCAL_IP}:9002'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
        time.sleep(4)

        if local_worker.poll() is not None:
            _, stderr = local_worker.communicate()
            print(f"  FAIL: {stderr.decode()[:500]}")
            return False
        print("  OK: Local worker started")

        # Step 4: Connect client
        print(f"[4/4] Connecting client to {REMOTE_IP}:8786...")
        signal.alarm(TIMEOUT)

        from distributed import Client
        client = Client(f'tcp://{REMOTE_IP}:8786', timeout=15)

        info = client.scheduler_info()
        n_workers = len(info['workers'])
        worker_info = [(w.get('name', 'unknown'), w.get('address', 'unknown'))
                       for w in info['workers'].values()]
        print(f"  OK: {n_workers} workers connected:")
        for name, addr in worker_info:
            print(f"      - {name}: {addr}")

        # Test basic computation
        print("\n[TEST 1] Basic computation...")
        import dask.array as da
        x = da.random.random((5000, 5000), chunks=(1000, 1000))
        result = x.sum().compute()
        print(f"  OK: sum = {result:.2f}")

        # Test work distribution
        print("\n[TEST 2] Checking work distribution...")
        import dask
        from dask import delayed

        @delayed
        def get_worker_id():
            import socket
            return socket.gethostname()

        tasks = [get_worker_id() for _ in range(10)]
        results = dask.compute(*tasks)
        unique_hosts = set(results)
        print(f"  OK: Work distributed across: {unique_hosts}")

        client.close()
        signal.alarm(0)

        print("\n" + "="*70)
        print("2-MACHINE TEST: SUCCESS")
        print("="*70)
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
        subprocess.run(['ssh', REMOTE_HOST, 'pkill -f dask; pkill -f distributed'],
                       capture_output=True, timeout=10)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
