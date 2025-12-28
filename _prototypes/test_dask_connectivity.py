#!/usr/bin/env python3
"""Dask connectivity diagnostic script with timeouts."""

import subprocess
import time
import signal
import sys
import socket

TIMEOUT = 10  # seconds per test
LOCAL_IP = "172.16.0.56"
REMOTE_HOST = "hvp-dev-mac2"
REMOTE_IP = "172.16.0.3"

def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out")

def run_test(name, test_func):
    """Run a test with timeout."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT)

    try:
        result = test_func()
        signal.alarm(0)
        print(f"RESULT: PASS - {result}")
        return True
    except TimeoutError:
        print(f"RESULT: TIMEOUT after {TIMEOUT}s")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"RESULT: FAIL - {e}")
        return False

def test_1_localhost_cluster():
    """Test LocalCluster on localhost."""
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=1, threads_per_worker=2, host='127.0.0.1')
    client = Client(cluster)
    n_workers = len(client.scheduler_info()['workers'])
    client.close()
    cluster.close()
    return f"{n_workers} workers"

def test_2_tcp_to_local_external_ip():
    """Test raw TCP connection to local external IP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    # First start a simple server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LOCAL_IP, 19999))
    server.listen(1)
    server.settimeout(5)

    # Connect from client
    s.connect((LOCAL_IP, 19999))
    conn, addr = server.accept()
    s.send(b"hello")
    data = conn.recv(1024)
    conn.close()
    s.close()
    server.close()
    return f"TCP works to {LOCAL_IP}"

def test_3_tcp_from_remote():
    """Test if remote can TCP connect to local."""
    import subprocess
    result = subprocess.run(
        ['ssh', REMOTE_HOST, f'nc -vz {LOCAL_IP} 22'],
        capture_output=True, text=True, timeout=10
    )
    if 'succeeded' in result.stderr.lower() or 'succeeded' in result.stdout.lower():
        return f"Remote can reach {LOCAL_IP}:22"
    return f"Remote connectivity: {result.stderr}"

def test_4_scheduler_client_localhost():
    """Start scheduler on localhost, connect client."""
    from distributed import Scheduler, Client
    import asyncio

    async def run():
        s = Scheduler(host='127.0.0.1', port=18786)
        await s.start()
        client = await Client(s.address, asynchronous=True)
        n = len(client.scheduler_info()['workers'])
        await client.close()
        await s.close()
        return n

    result = asyncio.run(run())
    return f"Client connected, {result} workers"

def test_5_scheduler_client_external_ip():
    """Start scheduler on external IP, connect client."""
    from distributed import Scheduler, Client
    import asyncio

    async def run():
        s = Scheduler(interface='en0', port=18787)
        await s.start()
        print(f"  Scheduler at: {s.address}")
        client = await Client(s.address, asynchronous=True)
        n = len(client.scheduler_info()['workers'])
        await client.close()
        await s.close()
        return n

    result = asyncio.run(run())
    return f"Client connected via external IP, {result} workers"

def test_6_scheduler_on_remote():
    """Start scheduler on remote, connect from local."""
    import subprocess

    # Start scheduler on remote
    ssh_cmd = f"cd /tmp/gbm-bench && /private/tmp/gbm-bench/.pixi/envs/default/bin/python -c \"from distributed import Scheduler; import asyncio; s = Scheduler(port=18788); asyncio.run(s.start()); print(f'Scheduler at: {{s.address}}'); import time; time.sleep(30)\""

    proc = subprocess.Popen(
        ['ssh', REMOTE_HOST, ssh_cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(5)

    # Try to connect
    from distributed import Client
    try:
        client = Client(f'tcp://{REMOTE_IP}:18788', timeout=5)
        n = len(client.scheduler_info()['workers'])
        client.close()
        result = f"Connected to remote scheduler, {n} workers"
    except Exception as e:
        result = f"Failed: {e}"
    finally:
        proc.terminate()

    return result

def main():
    print("Dask Connectivity Diagnostics")
    print(f"Local IP: {LOCAL_IP}")
    print(f"Remote: {REMOTE_HOST} ({REMOTE_IP})")
    print(f"Timeout per test: {TIMEOUT}s")

    tests = [
        ("LocalCluster on localhost", test_1_localhost_cluster),
        ("TCP to local external IP", test_2_tcp_to_local_external_ip),
        ("TCP from remote to local", test_3_tcp_from_remote),
        ("Scheduler+Client on localhost", test_4_scheduler_client_localhost),
        ("Scheduler+Client on external IP", test_5_scheduler_client_external_ip),
        ("Scheduler on remote, client local", test_6_scheduler_on_remote),
    ]

    results = []
    for name, func in tests:
        passed = run_test(name, func)
        results.append((name, passed))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

if __name__ == "__main__":
    main()
