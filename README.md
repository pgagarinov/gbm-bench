# GBM-Bench: XGBoost & CatBoost GPU Benchmark Suite

Comprehensive benchmark comparing XGBoost and CatBoost performance across CPU, single GPU, and multi-GPU configurations.

## Platform Support

This benchmark suite supports two platforms:

| Platform | CPU Benchmarks | GPU Benchmarks | Multi-GPU |
|----------|---------------|----------------|-----------|
| **Linux (x86_64)** | ✅ | ✅ | ✅ |
| **macOS (Apple Silicon)** | ✅ | ❌ | ❌ |

**Note**: GPU benchmarks require NVIDIA CUDA GPUs, which are only available on Linux. macOS with Apple Silicon can run CPU-only benchmarks.

## Hardware Requirements

### Linux (Full GPU Support)
- **CPU**: Multi-core processor (benchmark captures core count)
- **GPU**: NVIDIA GPU(s) with CUDA support
  - Single GPU minimum for GPU benchmarks
  - Multiple GPUs for multi-GPU benchmarks (tested with 4x NVIDIA L4)
- **RAM**: Sufficient for dataset size (default 10.5M samples requires ~8GB)
- **NVIDIA Driver**: 470+ recommended (tested with 570.133.20)

### macOS (CPU Only)
- **CPU**: Apple Silicon (M1/M2/M3) or Intel
- **RAM**: Sufficient for dataset size (default 10.5M samples requires ~8GB)
- Tested on Mac Studio with Apple M3 Ultra (32 cores, 512GB RAM)

## Software Requirements

- [Pixi](https://pixi.sh) package manager
- **Linux**: NVIDIA CUDA drivers installed system-wide
- **macOS**: No additional requirements (CPU-only)

## Installation

### 1. Install Pixi (if not already installed)

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2. Clone the repository

```bash
git clone git@github.com:pgagarinov/gbm-bench.git
cd gbm-bench
```

### 3. Install dependencies

```bash
pixi install
```

This installs all required packages. Dependencies vary by platform:

**Linux:**
- `xgboost>=2.1` with Dask support (from PyPI for CUDA support)
- `catboost>=1.2` (from conda)
- `dask-cuda>=24.0` for multi-GPU XGBoost
- `cuda-python>=13.1.1` for CUDA support
- `numpy`, `scikit-learn`, `tabulate`, `pyarrow`

**macOS:**
- `xgboost>=2.1` (from conda, CPU-only)
- `catboost>=1.2` (from PyPI to avoid numpy binary mismatch)
- `numpy`, `scikit-learn`, `tabulate`, `pyarrow`

## Running Benchmarks

### Linux: Full Benchmark (Recommended)

Run all 6 configurations (XGBoost/CatBoost x CPU/1GPU/4GPU):

```bash
pixi run python benchmark_full.py
```

### macOS: CPU-Only Benchmark

On macOS, skip GPU tests (they will fail without NVIDIA GPUs):

```bash
pixi run python benchmark_full.py --skip-1gpu --skip-multi-gpu
```

**Default parameters:**
- 10,500,000 samples
- 28 features
- 100 iterations
- Tree depth: 6
- Learning rate: 0.03

### Custom Parameters

```bash
# Fewer samples for quick test
pixi run python benchmark_full.py --samples 1000000

# More iterations
pixi run python benchmark_full.py --iterations 500

# Multiple iteration counts
pixi run python benchmark_full.py --iterations 100 500 1000

# Custom tree parameters
pixi run python benchmark_full.py --depth 8 --learning-rate 0.1

# Different number of GPUs for multi-GPU tests
pixi run python benchmark_full.py --n-gpus 2
```

### Single Test Mode

Run only a specific configuration (useful for debugging):

```bash
# Test XGBoost 4-GPU with Dask
pixi run python benchmark_full.py --test xgb-4gpu

# Test CatBoost 4-GPU
pixi run python benchmark_full.py --test cb-4gpu

# Available options: xgb-cpu, xgb-1gpu, xgb-4gpu, cb-cpu, cb-1gpu, cb-4gpu
```

### Quick Validation Test

```bash
# Small dataset to verify setup works
pixi run python benchmark_full.py --samples 100000 --iterations 10
```

## Output

### Console Output

Real-time progress showing:
- Configuration being tested
- Time elapsed
- Accuracy and AUC metrics
- GPU utilization statistics

### JSON Output

Results are saved to `benchmark_results_YYYYMMDD_HHMMSS.json` with:

```json
{
  "hardware": {
    "timestamp": "2025-12-23T20:11:26.347646",
    "platform": "Linux-6.8.0-90-generic-x86_64-with-glibc2.39",
    "python_version": "3.12.12",
    "cpu_count": 48,
    "cpu_model": "INTEL(R) XEON(R) PLATINUM 8558",
    "gpus": [
      {"name": "NVIDIA L4", "memory_mb": 24570, "driver": "570.133.20"}
    ],
    "gpu_count": 4,
    "xgboost_version": "3.1.2",
    "catboost_version": "1.2.8"
  },
  "parameters": {
    "samples": 10500000,
    "features": 28,
    "iterations": [100],
    "depth": 6,
    "learning_rate": 0.03
  },
  "benchmarks": [
    {
      "iterations": 100,
      "results": {
        "xgboost_cpu": {
          "time_seconds": 10.76,
          "accuracy": 0.9489,
          "auc": 0.9866,
          "gpu_stats": {}
        },
        "xgboost_1gpu": {
          "time_seconds": 5.99,
          "accuracy": 0.9487,
          "auc": 0.9865,
          "gpu_stats": {
            "gpu0_avg_util": 87.9,
            "gpu0_max_util": 100
          }
        },
        "xgboost_4gpu_dask": {
          "time_seconds": 4.26,
          "accuracy": 0.9493,
          "auc": 0.9867,
          "gpu_stats": {
            "gpu0_avg_util": 25.1,
            "gpu1_avg_util": 30.6,
            "gpu2_avg_util": 25.8,
            "gpu3_avg_util": 22.6
          }
        },
        "catboost_cpu": {...},
        "catboost_1gpu": {...},
        "catboost_4gpu": {...}
      }
    }
  ]
}
```

## Benchmark Scripts

| Script | Description |
|--------|-------------|
| `benchmark_full.py` | **Main script** - Full comparison with accuracy metrics, JSON output, Dask multi-GPU |
| `benchmark.py` | Basic benchmark with GPU monitoring and CLI flags |
| `benchmark_higgs.py` | CatBoost HIGGS dataset reproduction (from TowardsDataScience article) |
| `benchmark_higgs_xgb.py` | XGBoost HIGGS dataset benchmark |

## Multi-GPU Implementation Details

### XGBoost Multi-GPU

XGBoost uses **Dask** with `LocalCUDACluster` for multi-GPU training:
- Each GPU runs as a separate Dask worker
- Data is partitioned across GPUs using Dask arrays
- Requires `dask-cuda` package

### CatBoost Multi-GPU

CatBoost has **native multi-GPU support**:
- Uses `task_type="GPU"` with `devices="0-3"` syntax
- No additional distributed framework required

## Troubleshooting

### CUDA not available in XGBoost

The conda-forge XGBoost package doesn't include CUDA support. This repo uses PyPI installation:
```toml
[tool.pixi.pypi-dependencies]
xgboost = {version = ">=2.1", extras = ["dask"]}
```

### dask-cuda import errors

Ensure compatible versions:
```toml
numpy = ">=1.24,<2.3"  # dask-cuda requires numpy<2.3
cuda-python = ">=13.1.1,<14"
pyarrow = ">=22.0.0,<23"
```

### GPU not detected

1. Check NVIDIA driver: `nvidia-smi`
2. Verify CUDA is available in XGBoost:
   ```python
   import xgboost as xgb
   print(xgb.build_info())  # Should show CUDA: ON
   ```

### Permission denied on GPU

Ensure your user has access to NVIDIA devices:
```bash
ls -la /dev/nvidia*
```

## Reproducing Results

To reproduce the exact benchmark results on a different machine:

### Linux (Full GPU Benchmark)

```bash
# 1. Clone and install
git clone git@github.com:pgagarinov/gbm-bench.git
cd gbm-bench
pixi install

# 2. Run with same parameters as reference
pixi run python benchmark_full.py \
    --samples 10500000 \
    --features 28 \
    --iterations 100 \
    --depth 6 \
    --learning-rate 0.03 \
    --gpus 4

# 3. Compare your JSON output with reference files
```

### macOS (CPU-Only Benchmark)

```bash
# 1. Clone and install
git clone git@github.com:pgagarinov/gbm-bench.git
cd gbm-bench
pixi install

# 2. Run CPU-only benchmarks (skip GPU tests)
pixi run python benchmark_full.py \
    --samples 10500000 \
    --features 28 \
    --iterations 100 \
    --depth 6 \
    --learning-rate 0.03 \
    --skip-1gpu \
    --skip-multi-gpu

# 3. Compare your JSON output with reference files
```

**Note**: Results will vary based on hardware. The JSON captures your hardware configuration for comparison.

## Reference Results

### Linux: Intel Xeon Platinum 8558 (48 cores) + 4x NVIDIA L4 GPUs

| Configuration | Time (s) | Accuracy | AUC | GPU Speedup |
|---------------|----------|----------|-----|-------------|
| XGBoost CPU | 10.76 | 0.9489 | 0.9866 | - |
| XGBoost 1 GPU | 5.99 | 0.9487 | 0.9865 | 1.8x |
| XGBoost 4 GPU (Dask) | 4.26 | 0.9493 | 0.9867 | 2.5x |
| CatBoost CPU | 74.4 | 0.944 | 0.985 | - |
| CatBoost 1 GPU | 64.01 | 0.9444 | 0.9851 | 1.2x |
| CatBoost 4 GPU | 58.37 | 0.9437 | 0.9849 | 1.3x |

### macOS: Apple M3 Ultra (32 cores) - CPU Only

| Configuration | Time (s) | Accuracy | AUC |
|---------------|----------|----------|-----|
| XGBoost CPU | 6.85 | 0.9489 | 0.9866 |
| CatBoost CPU | 57.23 | 0.9442 | 0.9851 |

### CPU Performance Comparison (Xeon vs M3 Ultra)

| Test | Intel Xeon 8558 (48 cores) | Apple M3 Ultra (32 cores) | M3 Speedup |
|------|---------------------------|---------------------------|------------|
| XGBoost CPU | 10.76s | 6.85s | **1.57x faster** |
| CatBoost CPU | 74.4s | 57.23s | **1.30x faster** |

**Key findings:**
- XGBoost is ~7-8x faster than CatBoost on both platforms
- Both achieve similar accuracy (~0.94-0.95) and AUC (~0.985-0.987)
- Apple M3 Ultra outperforms 48-core Xeon on CPU benchmarks despite fewer cores
- GPU speedups on Linux are modest due to fast Xeon CPU and inference-optimized L4 GPUs

## License

MIT
