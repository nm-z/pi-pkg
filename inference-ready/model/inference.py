#!/usr/bin/env python3
"""
Inference script for the trained Custom ResNet model
Loads the model and makes predictions on new VNA feature rows
"""

import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
import joblib
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

# Version check utilities

def _compute_runtime_version() -> str:
    hasher = hashlib.sha256()
    try:
        with open('gpu_optuna_trainer.py', 'rb') as f:
            hasher.update(f.read())
    except Exception:
        pass
    return hasher.hexdigest()

RUNTIME_VERSION = _compute_runtime_version()

# Always prefer snapshot model if present for exact shape/params
_SNAPSHOT_MODEL = None
snap_model_path = Path('inference-ready/model/model_def.py')
if snap_model_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location('snapshot_model_def', snap_model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    _SNAPSHOT_MODEL = mod

# GPU-ONLY ENFORCEMENT
if not torch.cuda.is_available():
    raise RuntimeError("GPU NOT AVAILABLE - REQUIRES GPU FOR INFERENCE")
device = torch.device('cuda:0')
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

def _load_preprocessors(params: Dict) -> List[Tuple[str, object]]:
    """Load only enabled preprocessors, in the exact training order: var -> scaler -> kbest"""
    steps: List[Tuple[str, object]] = []
    base = 'inference-ready'
    if params.get('use_variance_threshold', False) and os.path.exists(f'{base}/var_thresh.pkl'):
        steps.append(('var', joblib.load(f'{base}/var_thresh.pkl')))
        print('Loaded VarianceThreshold')
    if params.get('use_standard_scaler', False) and os.path.exists(f'{base}/scaler.pkl'):
        steps.append(('scaler', joblib.load(f'{base}/scaler.pkl')))
        print('Loaded StandardScaler')
    if params.get('use_select_kbest', False) and os.path.exists(f'{base}/kbest_selector.pkl'):
        steps.append(('kbest', joblib.load(f'{base}/kbest_selector.pkl')))
        print('Loaded SelectKBest')
    return steps

def _apply_preprocessors(X: np.ndarray, steps: List[Tuple[str, object]]) -> np.ndarray:
    for name, tr in steps:
        X = tr.transform(X)
    return X

def load_model():
    """Load the trained model, parameters, and enabled preprocessing components.
    Determines correct input_dim by transforming a sample row if available."""
    # Load parameters and data stats
    with open('inference-ready/model_params.json', 'r') as f:
        params = json.load(f)
    with open('inference-ready/data_stats.json', 'r') as f:
        data_stats = json.load(f)

    # Enabled preprocessors in order
    preprocess_steps = _load_preprocessors(params)

    # Determine input_dim by transforming a sample from holdout if available
    input_dim = None
    # Prefer new holdout arrays; ignore stale X_test_raw from older runs
    x_sample_path = Path('inference-ready/X_holdout_raw.npy')
    if x_sample_path.exists():
        X_sample = np.load(x_sample_path)
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        else:
            X_sample = X_sample[:1]
        X_t = _apply_preprocessors(X_sample, preprocess_steps)
        input_dim = int(X_t.shape[1])
    else:
        # Fallbacks: prefer kbest_k if used, else raw X_shape[1]
        if params.get('use_select_kbest', False) and 'kbest_k' in params:
            input_dim = int(params['kbest_k'])
        else:
            input_dim = int(data_stats['X_shape'][1])

    # Create model from snapshot definition (matches training, includes residual_init)
    if _SNAPSHOT_MODEL is None:
        raise RuntimeError("Snapshot model not found at inference-ready/model/model_def.py")
    CustomResNet = _SNAPSHOT_MODEL.CustomResNet  # type: ignore

    model = CustomResNet(
        input_dim=input_dim,
        hidden_dim=params['hidden_dim'],
        num_blocks=params['num_blocks'],
        dropout=params['dropout'],
        residual_init=params.get('residual_init', 1.0),
    ).to(device)

    # Load trained weights
    model.load_state_dict(torch.load('inference-ready/resnet_model.pth', map_location=device))
    model.eval()

    return model, params, data_stats, preprocess_steps


def predict_from_raw(model: nn.Module, X_raw: np.ndarray, preprocess_steps: List[Tuple[str, object]]) -> np.ndarray:
    """Predict temperatures from raw VNA feature rows (shape: N x raw_dim)."""
    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(1, -1)
    X = _apply_preprocessors(X_raw, preprocess_steps)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(X_tensor)
    return preds.detach().cpu().numpy()


def main():
    print("Loading trained model and artifacts from inference-ready/...")
    model, params, data_stats, preprocess_steps = load_model()

    # Load metrics
    with open('inference-ready/metrics.json', 'r') as f:
        metrics = json.load(f)

    print("Model loaded successfully!")
    print(f"R² Score (training run): {metrics['r2_score']:.6f}")
    print(f"Model parameters: {params}")
    print(f"Preprocessing steps (enabled): {[name for name,_ in preprocess_steps]}")

    # Holdout evaluation using saved arrays (true end-to-end check)
    x_path = Path('inference-ready/X_holdout_raw.npy')
    y_path = Path('inference-ready/y_holdout.npy')
    if x_path.exists() and y_path.exists():
        X_test_raw = np.load(x_path)
        y_test = np.load(y_path)
        preds = predict_from_raw(model, X_test_raw, preprocess_steps)
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print("True Holdout Set Performance:")
        print(f"R²: {r2:.6f} | MAE: {mae:.4f}°C | RMSE: {rmse:.4f}°C")
        # Show a few examples with % accuracy
        print("Samples (first 10): actual vs predicted with % accuracy")
        print(f"{'idx':>4}  {'actual':>10}  {'pred':>10}  {'diff':>10}  {'acc%':>8}")
        for i in range(min(10, len(y_test))):
            actual = float(y_test[i])
            pred = float(preds[i])
            diff = abs(pred - actual)
            denom = max(1e-6, abs(actual))
            acc = max(0.0, 100.0 * (1.0 - diff / denom))
            print(f"{i:4d}  {actual:10.3f}  {pred:10.3f}  {diff:10.3f}  {acc:8.2f}")
    else:
        print("No holdout arrays found. Provide raw VNA feature rows (N x raw_dim) to predict_from_raw().")

    print("\nArtifacts available in inference-ready/: resnet_model.pth, model_params.json, metrics.json, data_stats.json, [var_thresh.pkl|scaler.pkl|kbest_selector.pkl], X_holdout_raw.npy, y_holdout.npy, model/model_def.py")

if __name__ == "__main__":
    main()
