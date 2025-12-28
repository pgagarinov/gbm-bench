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

## Scripts

**Main script:**
- `benchmark_2mac_xgboost.py` - XGBoost + Dask on 2-Mac cluster (with fix)

**Investigation scripts** preserved in `_prototypes/`:
- `test_dask_connectivity.py` - Connectivity diagnostics
- `test_approach1_ssh_tunnel.py` - SSH tunnel test
- `test_approach2_remote_scheduler.py` - Direct connection test
- `test_xgboost_dask.py` - XGBoost + Dask tests
- `test_xgboost_ray.py` - XGBoost + Ray tests
- `benchmark_*.py` - Various benchmark scripts
