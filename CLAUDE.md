# Claude Code Instructions

This is a GPU benchmark suite comparing XGBoost and CatBoost performance.

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
```
