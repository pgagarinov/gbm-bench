#!/usr/bin/env python3
"""
Distributed Random Forest Comparison: sklearn vs XGBoost vs LightGBM

Compares model-parallel RF distribution across three libraries.
Each worker trains a subset of trees on the full dataset (no sync needed).

Usage:
    python benchmark_rf_distributed.py
    python benchmark_rf_distributed.py --workers 4 --trees 200
    python benchmark_rf_distributed.py --samples 2000000
"""

import argparse
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from dask.distributed import Client, LocalCluster

# Default parameters
N_SAMPLES = 1_000_000
N_FEATURES = 50
N_TREES = 100
MAX_DEPTH = 8
N_WORKERS = 4


# =============================================================================
# Training functions (run on workers)
# =============================================================================

def train_sklearn_rf(X, y, n_trees, max_depth, seed):
    """Train sklearn RandomForest subset on worker."""
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X, y)
    return rf


def train_xgboost_rf(X, y, n_trees, max_depth, seed):
    """Train XGBoost RandomForest subset on worker."""
    import xgboost as xgb

    dtrain = xgb.DMatrix(X, label=y)

    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'max_depth': max_depth,
        'learning_rate': 1.0,  # No shrinkage for RF
        'subsample': 0.8,
        'colsample_bynode': 0.8,
        'num_parallel_tree': n_trees,
        'seed': seed,
        'nthread': -1,
    }

    model = xgb.train(params, dtrain, num_boost_round=1)
    return model


def train_lightgbm_rf(X, y, n_trees, max_depth, seed):
    """Train LightGBM RandomForest subset on worker."""
    import lightgbm as lgb

    # LightGBM RF uses boosting_type='rf' with bagging
    params = {
        'objective': 'binary',
        'boosting_type': 'rf',  # Random Forest mode
        'max_depth': max_depth,
        'num_leaves': 2 ** max_depth,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'feature_fraction': 0.8,
        'seed': seed,
        'num_threads': -1,
        'verbose': -1,
    }

    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=n_trees)
    return model


# =============================================================================
# Prediction functions
# =============================================================================

def predict_sklearn(models, X_test):
    """Combine predictions from sklearn RF models."""
    probas = np.zeros((len(X_test), len(models)))
    for i, rf in enumerate(models):
        probas[:, i] = rf.predict_proba(X_test)[:, 1]
    y_proba = probas.mean(axis=1)
    y_pred = (y_proba > 0.5).astype(int)
    return y_pred, y_proba


def predict_xgboost(models, X_test):
    """Combine predictions from XGBoost RF models."""
    import xgboost as xgb
    dtest = xgb.DMatrix(X_test)
    probas = np.zeros((len(X_test), len(models)))
    for i, model in enumerate(models):
        probas[:, i] = model.predict(dtest)
    y_proba = probas.mean(axis=1)
    y_pred = (y_proba > 0.5).astype(int)
    return y_pred, y_proba


def predict_lightgbm(models, X_test):
    """Combine predictions from LightGBM RF models."""
    probas = np.zeros((len(X_test), len(models)))
    for i, model in enumerate(models):
        probas[:, i] = model.predict(X_test)
    y_proba = probas.mean(axis=1)
    y_pred = (y_proba > 0.5).astype(int)
    return y_pred, y_proba


# =============================================================================
# Benchmark functions
# =============================================================================

def benchmark_single_machine(X_train, y_train, X_test, y_test, n_trees, max_depth):
    """Run single-machine baselines for all three libraries."""
    results = {}

    # sklearn
    print("\n  sklearn RF...", end=" ", flush=True)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                 random_state=42, n_jobs=-1)
    start = time.time()
    rf.fit(X_train, y_train)
    elapsed = time.time() - start
    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)
    results['sklearn'] = {
        'time': elapsed,
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
    print(f"{elapsed:.2f}s")

    # XGBoost
    print("  XGBoost RF...", end=" ", flush=True)
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'max_depth': max_depth,
        'learning_rate': 1.0,
        'subsample': 0.8,
        'colsample_bynode': 0.8,
        'num_parallel_tree': n_trees,
        'seed': 42,
        'nthread': -1,
    }
    start = time.time()
    model = xgb.train(params, dtrain, num_boost_round=1)
    elapsed = time.time() - start
    y_proba = model.predict(dtest)
    y_pred = (y_proba > 0.5).astype(int)
    results['xgboost'] = {
        'time': elapsed,
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
    print(f"{elapsed:.2f}s")

    # LightGBM
    print("  LightGBM RF...", end=" ", flush=True)
    import lightgbm as lgb
    params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'max_depth': max_depth,
        'num_leaves': 2 ** max_depth,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'feature_fraction': 0.8,
        'seed': 42,
        'num_threads': -1,
        'verbose': -1,
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    start = time.time()
    model = lgb.train(params, train_data, num_boost_round=n_trees)
    elapsed = time.time() - start
    y_proba = model.predict(X_test)
    y_pred = (y_proba > 0.5).astype(int)
    results['lightgbm'] = {
        'time': elapsed,
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
    print(f"{elapsed:.2f}s")

    return results


def benchmark_distributed(X_train, y_train, X_test, y_test,
                          n_trees, max_depth, n_workers, client):
    """Run model-parallel distributed RF for all three libraries."""
    results = {}
    trees_per_worker = n_trees // n_workers

    # Scatter data to all workers
    print(f"\n  Broadcasting data to {n_workers} workers...")
    X_future = client.scatter(X_train, broadcast=True)
    y_future = client.scatter(y_train, broadcast=True)

    # sklearn distributed
    print(f"  sklearn RF ({trees_per_worker} trees/worker)...", end=" ", flush=True)
    start = time.time()
    futures = [
        client.submit(train_sklearn_rf, X_future, y_future, trees_per_worker, max_depth, 42+i)
        for i in range(n_workers)
    ]
    models = client.gather(futures)
    elapsed = time.time() - start
    y_pred, y_proba = predict_sklearn(models, X_test)
    results['sklearn'] = {
        'time': elapsed,
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
    print(f"{elapsed:.2f}s")

    # XGBoost distributed
    print(f"  XGBoost RF ({trees_per_worker} trees/worker)...", end=" ", flush=True)
    start = time.time()
    futures = [
        client.submit(train_xgboost_rf, X_future, y_future, trees_per_worker, max_depth, 42+i)
        for i in range(n_workers)
    ]
    models = client.gather(futures)
    elapsed = time.time() - start
    y_pred, y_proba = predict_xgboost(models, X_test)
    results['xgboost'] = {
        'time': elapsed,
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
    print(f"{elapsed:.2f}s")

    # LightGBM distributed
    print(f"  LightGBM RF ({trees_per_worker} trees/worker)...", end=" ", flush=True)
    start = time.time()
    futures = [
        client.submit(train_lightgbm_rf, X_future, y_future, trees_per_worker, max_depth, 42+i)
        for i in range(n_workers)
    ]
    models = client.gather(futures)
    elapsed = time.time() - start
    y_pred, y_proba = predict_lightgbm(models, X_test)
    results['lightgbm'] = {
        'time': elapsed,
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
    print(f"{elapsed:.2f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Distributed RF Comparison")
    parser.add_argument("--samples", type=int, default=N_SAMPLES, help="Number of samples")
    parser.add_argument("--features", type=int, default=N_FEATURES, help="Number of features")
    parser.add_argument("--trees", type=int, default=N_TREES, help="Number of trees")
    parser.add_argument("--depth", type=int, default=MAX_DEPTH, help="Max tree depth")
    parser.add_argument("--workers", type=int, default=N_WORKERS, help="Number of Dask workers")
    args = parser.parse_args()

    print("=" * 70)
    print("Distributed Random Forest: sklearn vs XGBoost vs LightGBM")
    print("=" * 70)
    print(f"Samples: {args.samples:,}")
    print(f"Features: {args.features}")
    print(f"Trees: {args.trees}")
    print(f"Max Depth: {args.depth}")
    print(f"Workers: {args.workers}")

    # Generate data
    print("\nGenerating data...")
    X, y = make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=args.features // 2,
        n_redundant=args.features // 4,
        random_state=42
    )
    X = X.astype(np.float32)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Single machine baselines
    print("\n" + "-" * 70)
    print("SINGLE MACHINE (baseline)")
    print("-" * 70)
    single_results = benchmark_single_machine(
        X_train, y_train, X_test, y_test, args.trees, args.depth
    )

    # Create Dask cluster
    print("\n" + "-" * 70)
    print(f"DISTRIBUTED ({args.workers} workers, model-parallel)")
    print("-" * 70)
    cluster = LocalCluster(
        n_workers=args.workers,
        threads_per_worker=max(1, 8 // args.workers)
    )
    client = Client(cluster)

    dist_results = benchmark_distributed(
        X_train, y_train, X_test, y_test,
        args.trees, args.depth, args.workers, client
    )

    client.close()
    cluster.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Library':<12} {'Single':>10} {'Distributed':>12} {'Speedup':>10} {'AUC (S/D)':>15}")
    print("-" * 70)

    for lib in ['sklearn', 'xgboost', 'lightgbm']:
        s = single_results[lib]
        d = dist_results[lib]
        speedup = s['time'] / d['time']
        print(f"{lib:<12} {s['time']:>9.2f}s {d['time']:>11.2f}s {speedup:>9.2f}x "
              f"{s['auc']:>7.4f}/{d['auc']:.4f}")

    # Find best
    print("\n" + "-" * 70)
    fastest_single = min(single_results.items(), key=lambda x: x[1]['time'])
    fastest_dist = min(dist_results.items(), key=lambda x: x[1]['time'])
    best_auc_single = max(single_results.items(), key=lambda x: x[1]['auc'])

    print(f"Fastest single-machine: {fastest_single[0]} ({fastest_single[1]['time']:.2f}s)")
    print(f"Fastest distributed:    {fastest_dist[0]} ({fastest_dist[1]['time']:.2f}s)")
    print(f"Best AUC:               {best_auc_single[0]} ({best_auc_single[1]['auc']:.4f})")

    # Scaling analysis
    print("\n" + "-" * 70)
    print("SCALING ANALYSIS")
    print("-" * 70)
    print(f"\n{'Library':<12} {'Efficiency':>12} {'Note':<40}")
    print("-" * 70)
    for lib in ['sklearn', 'xgboost', 'lightgbm']:
        s = single_results[lib]
        d = dist_results[lib]
        speedup = s['time'] / d['time']
        efficiency = speedup / args.workers * 100

        if efficiency > 80:
            note = "Excellent scaling"
        elif efficiency > 50:
            note = "Good scaling"
        elif efficiency > 25:
            note = "Limited by single-machine parallelism"
        else:
            note = "Overhead dominates"

        print(f"{lib:<12} {efficiency:>10.0f}% {note:<40}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. All three libraries can be distributed using model-parallel approach
2. Each worker trains subset of trees on FULL data (no histogram sync)
3. On single machine, scaling limited because libraries already use all cores
4. Real benefit on MULTI-MACHINE clusters where each machine is independent
5. XGBoost/LightGBM RF often faster but may have different accuracy profile
""")

    return 0


if __name__ == "__main__":
    exit(main())
