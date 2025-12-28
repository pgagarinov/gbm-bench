#!/usr/bin/env python3
"""
Model-Parallel Random Forest with Dask.

Demonstrates the difference between:
1. Data-parallel (XGBoost Dask): Distributes DATA, syncs histograms each iteration
2. Model-parallel (this script): Distributes TREES, no sync needed

Model-parallel RF scales much better because RF trees are independent
(embarrassingly parallel) - no synchronization required between workers.

Usage:
    python benchmark_rf_model_parallel.py
    python benchmark_rf_model_parallel.py --workers 4 --trees 200
    python benchmark_rf_model_parallel.py --samples 2000000
"""

import argparse
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from dask.distributed import Client, LocalCluster

# Default parameters
N_SAMPLES = 1_000_000
N_FEATURES = 50
N_TREES = 100
MAX_DEPTH = 8
N_WORKERS = 4


def train_rf_subset(X, y, n_trees, max_depth, seed):
    """
    Train a subset of RF trees on a worker.

    This function runs on each Dask worker independently.
    Each worker trains n_trees trees on the FULL dataset.
    No synchronization needed between workers.
    """
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1  # Use all cores on this worker
    )
    rf.fit(X, y)
    return rf


def combine_predictions(forests, X_test, method='vote'):
    """
    Combine predictions from multiple RF models.

    Args:
        forests: List of trained RandomForestClassifier models
        X_test: Test data
        method: 'vote' for majority voting, 'proba' for probability averaging
    """
    if method == 'proba':
        # Average probabilities
        probas = np.zeros((len(X_test), 2))
        for rf in forests:
            probas += rf.predict_proba(X_test)
        probas /= len(forests)
        return (probas[:, 1] > 0.5).astype(int), probas[:, 1]
    else:
        # Majority voting
        predictions = np.column_stack([rf.predict(X_test) for rf in forests])
        y_pred = (predictions.mean(axis=1) > 0.5).astype(int)
        # For AUC, average probabilities
        probas = np.column_stack([rf.predict_proba(X_test)[:, 1] for rf in forests])
        y_proba = probas.mean(axis=1)
        return y_pred, y_proba


def benchmark_sklearn_rf(X_train, y_train, X_test, y_test, n_trees, max_depth):
    """Baseline: sklearn RF using local threads."""
    print(f"\n[1] sklearn RF (local, all cores)")
    print("-" * 50)

    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    start = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"    Time: {train_time:.2f}s")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    AUC: {auc:.4f}")

    return {"time": train_time, "accuracy": accuracy, "auc": auc}


def benchmark_model_parallel_rf(X_train, y_train, X_test, y_test,
                                 n_trees, max_depth, n_workers, client):
    """
    Model-parallel RF: Distribute trees across Dask workers.

    Each worker trains a subset of trees on the full dataset.
    No synchronization needed during training.
    """
    print(f"\n[2] Model-Parallel RF ({n_workers} workers)")
    print("-" * 50)

    trees_per_worker = n_trees // n_workers
    print(f"    Trees per worker: {trees_per_worker}")

    # Scatter data to all workers (broadcast)
    print("    Broadcasting data to workers...")
    X_future = client.scatter(X_train, broadcast=True)
    y_future = client.scatter(y_train, broadcast=True)

    # Submit tree training to workers in parallel
    print("    Training trees in parallel...")
    start = time.time()

    futures = [
        client.submit(
            train_rf_subset,
            X_future, y_future,
            trees_per_worker, max_depth,
            seed=42 + i
        )
        for i in range(n_workers)
    ]

    # Gather trained forests from all workers
    forests = client.gather(futures)
    train_time = time.time() - start

    # Combine predictions
    print("    Combining predictions...")
    y_pred, y_proba = combine_predictions(forests, X_test, method='proba')

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Count total trees
    total_trees = sum(len(rf.estimators_) for rf in forests)

    print(f"    Time: {train_time:.2f}s")
    print(f"    Total trees: {total_trees}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    AUC: {auc:.4f}")

    return {"time": train_time, "accuracy": accuracy, "auc": auc, "total_trees": total_trees}


def benchmark_xgboost_dask_rf(X_train, y_train, X_test, y_test,
                               n_trees, max_depth, client):
    """
    XGBoost RF with Dask (data-parallel for comparison).
    """
    import xgboost as xgb
    import xgboost.dask as dxgb
    from xgboost.collective import Config
    import dask.array as da

    print(f"\n[3] XGBoost RF + Dask (data-parallel)")
    print("-" * 50)

    n_workers = len(client.scheduler_info()['workers'])
    chunk_size = max(1, len(X_train) // (n_workers * 2))

    X_da = da.from_array(X_train, chunks=(chunk_size, -1))
    y_da = da.from_array(y_train, chunks=(chunk_size,))

    dtrain = dxgb.DaskDMatrix(client, X_da, y_da)
    dtest = xgb.DMatrix(X_test)

    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'max_depth': max_depth,
        'learning_rate': 1.0,  # No shrinkage for RF
        'subsample': 0.8,
        'colsample_bynode': 0.8,
        'num_parallel_tree': n_trees,  # RF-style parallel trees
    }

    # Key fix for macOS
    coll_cfg = Config(retry=3, timeout=60, tracker_host_ip='127.0.0.1', tracker_port=0)

    print("    Training...")
    start = time.time()
    output = dxgb.train(client, params, dtrain, num_boost_round=1, coll_cfg=coll_cfg)
    train_time = time.time() - start

    model = output['booster']
    y_proba = model.predict(dtest)
    y_pred = (y_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"    Time: {train_time:.2f}s")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    AUC: {auc:.4f}")

    return {"time": train_time, "accuracy": accuracy, "auc": auc}


def main():
    parser = argparse.ArgumentParser(description="Model-Parallel RF Benchmark")
    parser.add_argument("--samples", type=int, default=N_SAMPLES, help="Number of samples")
    parser.add_argument("--features", type=int, default=N_FEATURES, help="Number of features")
    parser.add_argument("--trees", type=int, default=N_TREES, help="Number of trees")
    parser.add_argument("--depth", type=int, default=MAX_DEPTH, help="Max tree depth")
    parser.add_argument("--workers", type=int, default=N_WORKERS, help="Number of Dask workers")
    parser.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost comparison")
    args = parser.parse_args()

    print("=" * 60)
    print("Model-Parallel vs Data-Parallel Random Forest")
    print("=" * 60)
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

    results = {}

    # Baseline: sklearn RF
    results['sklearn'] = benchmark_sklearn_rf(
        X_train, y_train, X_test, y_test, args.trees, args.depth
    )

    # Create Dask cluster
    print(f"\nCreating Dask cluster with {args.workers} workers...")
    cluster = LocalCluster(
        n_workers=args.workers,
        threads_per_worker=max(1, 8 // args.workers)
    )
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")

    # Model-parallel RF
    results['model_parallel'] = benchmark_model_parallel_rf(
        X_train, y_train, X_test, y_test,
        args.trees, args.depth, args.workers, client
    )

    # XGBoost Dask RF (data-parallel)
    if not args.skip_xgboost:
        results['xgb_dask'] = benchmark_xgboost_dask_rf(
            X_train, y_train, X_test, y_test,
            args.trees, args.depth, client
        )

    client.close()
    cluster.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    baseline_time = results['sklearn']['time']

    print(f"\n{'Method':<30} {'Time':>8} {'Speedup':>10} {'AUC':>8}")
    print("-" * 60)

    for name, res in results.items():
        speedup = baseline_time / res['time']
        label = {
            'sklearn': 'sklearn RF (baseline)',
            'model_parallel': f'Model-parallel RF ({args.workers}w)',
            'xgb_dask': f'XGBoost Dask RF ({args.workers}w)'
        }.get(name, name)
        print(f"{label:<30} {res['time']:>7.2f}s {speedup:>9.2f}x {res['auc']:>8.4f}")

    print("\n" + "-" * 60)
    print("Key insight:")
    print("  Model-parallel (distribute trees) >> Data-parallel (distribute data)")
    print("  RF trees are independent - no sync needed between workers")
    print("  XGBoost Dask syncs histograms every iteration - high overhead")

    return 0


if __name__ == "__main__":
    exit(main())
