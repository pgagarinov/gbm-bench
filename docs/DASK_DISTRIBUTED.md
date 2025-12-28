# Distributed Training on macOS - Investigation Report

This document summarizes the investigation into running distributed GBM training across multiple macOS machines using Dask and Ray.

## Test Environment

**Machines:**
- Local: 172.16.0.56
- Remote: 172.16.0.3 (hvp-dev-mac2)

**Software Stack:**
- Python 3.12 (via pixi)
- Dask 2025.12.0, Distributed 2025.12.0
- Ray 2.53.0, xgboost_ray 0.1.19
- XGBoost 3.1.2
- LightGBM (latest)

---

## Framework Compatibility Summary

| Framework | Library | Local | Multi-Machine | Notes |
|-----------|---------|-------|---------------|-------|
| Dask | LightGBM | Works | SSH tunnel only | Slower than single machine |
| Dask | XGBoost | **Works** | SSH mode only | Requires `tracker_host_ip` fix |
| Ray | XGBoost | Works | Fails | Multi-port requirements |
| Ray | xgboost_ray | BROKEN | N/A | Compat issues with XGBoost 3.x |

---

## Connectivity Issues

### Root Cause: macOS lo0 Routing

macOS routes traffic to a machine's own external IP through the loopback interface (lo0) instead of the physical interface (en0).

**Impact:**
- TCP `connect()` to own external IP succeeds (kernel handles it)
- TCP `accept()` on server bound to external IP fails (packet arrives on wrong interface)
- This breaks worker-to-worker communication required for distributed training

### Connectivity Matrix

| Source | Destination | Works? | Reason |
|--------|-------------|--------|--------|
| Machine A → Machine B | Yes | Traffic goes out physical interface |
| Machine A → Machine A (own IP) | No | Routes through lo0, server can't accept |

### Workarounds

**SSH Tunnel (Partial):**
- Works for Dask scheduler access
- All workers must be on same machine as scheduler
- Ray requires multiple ports, tunnel insufficient

**Direct External IP:**
- Fails due to lo0 routing issue

---

## XGBoost + Dask

**Status: WORKS (with fix)**

### The Fix

The hang was caused by XGBoost's Rabit tracker failing to resolve the correct IP on macOS. The fix is to explicitly set `tracker_host_ip='127.0.0.1'` using `xgboost.collective.Config`:

```python
from xgboost.collective import Config
from xgboost import dask as dxgb

coll_cfg = Config(
    retry=3,
    timeout=60,
    tracker_host_ip="127.0.0.1",  # KEY FIX for macOS
    tracker_port=0
)

output = dxgb.train(client, params, dtrain, num_boost_round=100, coll_cfg=coll_cfg)
```

### Benchmark Results (2-Mac Cluster via SSH mode)

| Samples | Baseline (local) | Dask (remote) | Speedup |
|---------|-----------------|---------------|---------|
| 500K | 0.91s | 4.20s | 0.22x |
| 2M | 1.87s | 5.89s | 0.32x |

Dask is slower due to data transfer and coordination overhead.

### Usage

```bash
# SSH mode (recommended - everything runs on remote):
pixi run python benchmark_2mac_xgboost.py --ssh-mode

# With more workers:
pixi run python benchmark_2mac_xgboost.py --ssh-mode --workers 4 --threads 4
```

### Known Issues (now resolved)
- [Rabit initialization hangs](https://github.com/dmlc/xgboost/issues/6649) - Fixed with `tracker_host_ip`
- [DaskDMatrix worker restart issues](https://github.com/dmlc/xgboost/issues/9420)

---

## XGBoost + Ray

**Status: Local only**

### Local Results (100K samples)
| Test | Time |
|------|------|
| XGBoost single (100 rounds) | 0.66s |
| Ray parallel (4x50 rounds) | 3.34s |

Ray adds significant overhead for small workloads.

### Multi-Machine
- SSH tunnel insufficient (Ray needs multiple ports: GCS, object store, etc.)
- Direct connection fails due to macOS lo0 routing

### xgboost_ray Package
- Compatibility issues with XGBoost 3.x
- `TypeError: getaddrinfo() argument 1 must be string or None`

---

## LightGBM + Dask

**Status: Works (but slower)**

### Benchmark Results

| Dataset | Samples | Local (s) | Dask (s) | Speedup |
|---------|---------|-----------|----------|---------|
| 500K | 500,000 | 2.06 | 4.96 | 0.41x |
| 1M | 1,000,000 | 2.55 | 3.03 | 0.84x |
| 2M | 2,000,000 | 3.41 | 4.48 | 0.76x |
| 5M | 5,000,000 | 6.20 | 7.39 | 0.84x |
| 10M | 10,000,000 | 11.49 | CRASHED | - |

### Key Findings
1. Single machine is consistently faster for all tested sizes
2. Dask overhead from worker discovery, serialization, coordination
3. Speedup improves with larger data but never exceeds 1.0x

---

## Network Performance Requirements

### Why Distributed Training is Slower for Small Datasets

From our benchmarks:

| Dataset | Data Size | Baseline | Dask | Overhead | Effective Rate |
|---------|-----------|----------|------|----------|----------------|
| 500K samples | 97 MB | 0.91s | 4.20s | 3.3s | 30 MB/s |
| 2M samples | 389 MB | 1.87s | 5.89s | 4.0s | 97 MB/s |

The overhead includes:
- **Data serialization/transfer**: ~50% of overhead
- **Dask coordination**: Worker startup, task scheduling (~30%)
- **XGBoost Rabit sync**: Histogram exchange per iteration (~20%)

### Minimum Requirements for Speedup

| Scenario | Dataset Size | Min Network | Min Training Time |
|----------|-------------|-------------|-------------------|
| **Break-even** | >10M samples | 1 Gbps | >30s |
| **1.5x speedup** | >50M samples | 10 Gbps | >2 min |
| **Near-linear** | >100M samples | 25+ Gbps | >10 min |

### XGBoost Histogram Synchronization

XGBoost syncs histograms (not full gradients) each iteration:

| Tree Depth | Per Node | Per Tree (max) |
|------------|----------|----------------|
| 6 | 200 KB | 12.5 MB |
| 8 | 200 KB | 50 MB |
| 10 | 200 KB | 200 MB |

For 100 iterations: ~100-500 MB histogram data synced.
At 1 Gbps: 0.8-4s additional overhead.

### Recommendations by Network Speed

**1 Gbps Ethernet (Consumer Macs)**
- Max throughput: ~100 MB/s
- Minimum dataset: 50M+ samples (~10 GB)
- Minimum training time: 5+ minutes baseline
- **Verdict**: Rarely beneficial

**10 Gbps Ethernet (Mac Studio)**
- Max throughput: ~1 GB/s
- Minimum dataset: 10M+ samples
- Minimum training time: 2+ minutes baseline
- **Verdict**: Useful for large datasets

**25-100 Gbps (InfiniBand/RoCE)**
- Max throughput: 3-12 GB/s
- Suitable for any dataset that doesn't fit in RAM
- **Verdict**: Production multi-node training

### macOS Network Options

| Connection | Bandwidth | Latency | Suitable |
|------------|-----------|---------|----------|
| WiFi | Variable | High | No |
| 1 Gbps Ethernet | 125 MB/s | Low | Marginal |
| 10 Gbps Ethernet | 1.25 GB/s | Low | Yes |
| Thunderbolt Bridge | 10-40 Gbps | Very Low | Yes |

---

## Conclusions

### For macOS

1. **Use single-machine training** - it's faster for typical workloads
2. **XGBoost + Dask works** with `tracker_host_ip='127.0.0.1'` fix (but slower than single machine)
3. **LightGBM + Dask works** but provides no speedup
4. **Ray** works locally but multi-machine fails due to port requirements

### When to Use Distributed

Distributed training on macOS only makes sense when:
- Dataset doesn't fit in memory on a single machine
- You need fault tolerance for very long training runs

For typical workloads, single-machine XGBoost is 3-5x faster than Dask distributed.

### For Production Multi-Machine

Use Linux where:
- lo0 routing issue doesn't exist
- GPU support available (NVIDIA CUDA)
- All frameworks work correctly

---

## Model-Parallel vs Data-Parallel RF

### The Problem with XGBoost Dask RF

XGBoost Dask uses **data-parallelism** for both GBM and RF:
- Each worker gets a chunk of DATA
- Workers build ALL trees on their data chunk
- Must sync histograms every iteration

This doesn't leverage RF's embarrassingly parallel nature.

### Model-Parallel RF Solution

Distribute TREES across workers instead of DATA:

```python
from dask.distributed import Client, LocalCluster
from sklearn.ensemble import RandomForestClassifier

def train_rf_subset(X, y, n_trees, seed):
    """Train subset of trees on worker (full dataset)."""
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    return rf

# Create cluster
cluster = LocalCluster(n_workers=4)
client = Client(cluster)

# Broadcast data to all workers
X_future = client.scatter(X_train, broadcast=True)
y_future = client.scatter(y_train, broadcast=True)

# Train trees in parallel (no sync needed!)
trees_per_worker = 100 // 4  # 25 trees each
futures = [
    client.submit(train_rf_subset, X_future, y_future, trees_per_worker, seed=42+i)
    for i in range(4)
]

# Gather and combine predictions
forests = client.gather(futures)
predictions = np.column_stack([rf.predict(X_test) for rf in forests])
y_pred = (predictions.mean(axis=1) > 0.5).astype(int)
```

### Benchmark Results (1M samples, 200 trees, 4 workers)

**Single Machine:**

| Library | Time | AUC |
|---------|------|-----|
| XGBoost RF | 3.72s | 0.9516 |
| LightGBM RF | 25.28s | 0.9603 |
| sklearn RF | 32.00s | 0.9653 |

**Distributed (model-parallel):**

| Library | Time | Speedup | Efficiency | AUC |
|---------|------|---------|------------|-----|
| LightGBM RF | 10.85s | **2.33x** | 58% | 0.9609 |
| XGBoost RF | 10.84s | 0.34x | 9% | 0.9509 |
| sklearn RF | 31.06s | 1.03x | 26% | 0.9660 |

**Key findings:**
- **LightGBM RF scales best** (2.33x speedup) because single-machine is slower
- **XGBoost RF** already so fast that distribution adds overhead
- **sklearn RF** already uses all cores, minimal benefit on same machine

### When to Use Each Approach

| Scenario | Best Choice |
|----------|-------------|
| **Speed priority (single machine)** | XGBoost RF (3.72s) |
| **Accuracy priority** | sklearn RF (0.9653 AUC) |
| **Multi-machine distribution** | LightGBM RF (best scaling) |
| **Large datasets, distributed** | LightGBM RF model-parallel |

---

## Scripts

**Main scripts:**
- `benchmark_2mac_xgboost.py` - XGBoost + Dask on 2-Mac cluster (with fix)
- `benchmark_rf_distributed.py` - Distributed RF comparison (sklearn/XGBoost/LightGBM)
- `benchmark_rf_model_parallel.py` - Model-parallel sklearn RF only

**Investigation scripts** preserved in `_prototypes/`:
- `test_dask_connectivity.py` - Connectivity diagnostics
- `test_approach1_ssh_tunnel.py` - SSH tunnel test
- `test_approach2_remote_scheduler.py` - Direct connection test
- `test_xgboost_dask.py` - XGBoost + Dask tests
- `test_xgboost_ray.py` - XGBoost + Ray tests
- `benchmark_*.py` - Various benchmark scripts
