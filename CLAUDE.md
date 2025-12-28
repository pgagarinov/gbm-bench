# Claude Code Instructions

This is a GPU benchmark suite comparing XGBoost, CatBoost, and LightGBM performance.

## Project Overview

See [README.md](README.md) for complete documentation including:
- Installation instructions
- How to run benchmarks
- CLI options and parameters
- Output format specification
- Troubleshooting guide

## Key Files

- `benchmark_full.py` - Main benchmark script with all features
- `pyproject.toml` - Dependencies managed via pixi
- `benchmark_results*.json` - Saved results with hardware config

## Platform Support

- **Linux (x86_64)**: Full GPU + CPU benchmarks
- **macOS (Apple Silicon)**: CPU-only benchmarks (no NVIDIA CUDA)

## Quick Commands

```bash
pixi install                                    # Install dependencies
pixi run python benchmark_full.py               # Run full benchmark (Linux)
pixi run python benchmark_full.py --skip-1gpu --skip-multi-gpu  # CPU only (macOS)
pixi run python benchmark_full.py --test xgb-4gpu  # Test single config (Linux)
pixi run python benchmark_full.py --test xgb-cpu   # Test XGBoost CPU
pixi run python benchmark_full.py --test cb-cpu    # Test CatBoost CPU
pixi run python benchmark_full.py --test lgb-cpu   # Test LightGBM CPU
pixi run python benchmark_full.py --test lgb-4gpu  # Test LightGBM multi-GPU (native)
pixi run python benchmark_full.py --test lgb-4gpu-dask  # Test LightGBM multi-GPU (Dask)
pixi run python benchmark_full.py --test rf-sklearn   # Test sklearn RandomForest
pixi run python benchmark_full.py --test rf-xgboost   # Test XGBoost RandomForest
pixi run python benchmark_full.py --test rf-lightgbm  # Test LightGBM RandomForest
pixi run python benchmark_full.py --skip-rf           # Skip RF benchmarks

# 2-Mac cluster (XGBoost + Dask)
pixi run python benchmark_2mac_xgboost.py              # Workers on remote, tracker on local
pixi run python benchmark_2mac_xgboost.py --ssh-mode   # Everything on remote (most reliable)
pixi run python benchmark_2mac_xgboost.py --debug      # With debug output

# Model-parallel RF (distributes trees, not data)
pixi run python benchmark_rf_model_parallel.py                    # Default: 1M samples, 100 trees
pixi run python benchmark_rf_model_parallel.py --workers 4        # 4 Dask workers
pixi run python benchmark_rf_model_parallel.py --trees 200        # More trees
```

## Multi-Machine Distributed Training (macOS)

See `docs/DASK_DISTRIBUTED.md` for detailed investigation.

**XGBoost + Dask on 2-Mac cluster:**
- Use `benchmark_2mac_xgboost.py` with two modes:
  - Default: Workers on remote, XGBoost tracker on local
  - SSH mode (`--ssh-mode`): Everything runs on remote (avoids lo0 issue)
- The macOS lo0 routing issue prevents workers and tracker on same machine
- See script comments for details on the networking constraints
