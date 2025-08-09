#!/usr/bin/env python3
"""
GPU-Only Optuna Trainer for Custom ResNet
Trains until R² > 0.98 and saves inference artifacts
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as skl_r2
try:
    from torchmetrics.functional import r2_score as gpu_r2
    _HAS_TORCHMETRICS = True
except Exception:
    _HAS_TORCHMETRICS = False
    def gpu_r2(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Fallback GPU-side R² without torchmetrics
        yt = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0).float()
        yp = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0).float()
        mean = yt.mean()
        ss_res = torch.sum((yt - yp) ** 2)
        ss_tot = torch.sum((yt - mean) ** 2) + 1e-12
        return 1.0 - ss_res / ss_tot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
import pickle
import json
from pathlib import Path
import warnings
import joblib
warnings.filterwarnings('ignore')
from datetime import datetime
import glob
import hashlib
import shutil
import io
import sys
from tqdm import tqdm
import subprocess
import shlex

# Rich live display
from rich.live import Live

# AMP config
USE_AMP = True

# Unified R² helper (returns sklearn R² and GPU R²)
def r2_both(y_true_t: torch.Tensor, y_pred_t: torch.Tensor) -> tuple[float, float]:
    yt = torch.nan_to_num(y_true_t, nan=0.0, posinf=0.0, neginf=0.0)
    yp = torch.nan_to_num(y_pred_t, nan=0.0, posinf=0.0, neginf=0.0)
    # GPU metric
    try:
        r2_gpu = float(gpu_r2(yp, yt).detach().item())
    except Exception:
        mean = yt.mean()
        ss_res = torch.sum((yt - yp) ** 2)
        ss_tot = torch.sum((yt - mean) ** 2) + 1e-12
        r2_gpu = float((1.0 - ss_res / ss_tot).detach().item())
    # CPU sklearn for Optuna objective
    yt_np = yt.detach().cpu().numpy()
    yp_np = yp.detach().cpu().numpy()
    m = np.isfinite(yt_np) & np.isfinite(yp_np)
    r2_skl = float(skl_r2(yt_np[m], yp_np[m])) if m.sum() >= 2 else float("-inf")
    return r2_skl, r2_gpu

# Compile flag (set False on ROCm/HIP)
USE_COMPILE = False
try:
    if getattr(torch.version, 'hip', None):
        USE_COMPILE = False
    else:
        USE_COMPILE = True
except Exception:
    USE_COMPILE = False

# GPU caching toggle
CACHE_ON_GPU = True

# Toggle for GPU util logging (disabled per user preference)
LOG_GPU_UTIL = False

def _cache_arrays_to_gpu(X: np.ndarray, y: np.ndarray, device):
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device).contiguous()
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device).contiguous()
    bytes_X = X_t.numel() * 4
    bytes_y = y_t.numel() * 4
    print(f"Cached to GPU: X={bytes_X/1024/1024:.2f} MiB, y={bytes_y/1024/1024:.2f} MiB")
    return X_t, y_t

def _train_eval_epoch_cached(model, X_train_t, y_train_t, X_test_t, y_test_t, batch_size, criterion, optimizer, scaler, epoch, use_amp):
    # Train
    model.train()
    n = X_train_t.shape[0]
    perm = torch.randperm(n, device=X_train_t.device)
    train_loss_total = 0.0
    num_batches = 0
    for start in range(0, n, batch_size):
        idx = perm[start:start+batch_size]
        batch_X = X_train_t.index_select(0, idx)
        batch_y = y_train_t.index_select(0, idx)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss_total += float(loss.item())
        num_batches += 1
    avg_train_loss = train_loss_total / max(1, num_batches)
    # Eval
    model.eval()
    preds = []
    targets = []
    m = X_test_t.shape[0]
    with torch.no_grad():
        for start in range(0, m, batch_size):
            batch_X = X_test_t[start:start+batch_size]
            batch_y = y_test_t[start:start+batch_size]
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                out = model(batch_X)
            preds.append(out.float())
            targets.append(batch_y.float())
    preds_t = torch.cat(preds, dim=0)
    targets_t = torch.cat(targets, dim=0)
    r2_skl, _ = r2_both(targets_t, preds_t)
    return avg_train_loss, r2_skl

# Live one-line writer
_live_prev_len = 0

def _live_update(msg: str):
    global _live_prev_len
    s = f"\r{msg}"
    pad = max(0, _live_prev_len - len(msg))
    sys.stdout.write(s + (" " * pad))
    sys.stdout.flush()
    _live_prev_len = len(msg)

def _live_newline():
    global _live_prev_len
    sys.stdout.write("\n")
    sys.stdout.flush()
    _live_prev_len = 0

# Make rocm-smi non-blocking and auto-disable after first failure
_GPU_SMI_ENABLED = False  # start disabled

def _log_gpu_util(note: str = ""):
    if not LOG_GPU_UTIL:
        return
    try:
        mem_alloc = torch.cuda.memory_allocated() / (1024**3)
        mem_reserved = torch.cuda.memory_reserved() / (1024**3)
        msg = f"GPU mem: allocated={mem_alloc:.2f} GiB reserved={mem_reserved:.2f} GiB"
        if note:
            msg = f"[{note}] " + msg
        _live_update(msg)
    except Exception:
        pass
    global _GPU_SMI_ENABLED
    if not _GPU_SMI_ENABLED:
        return
    try:
        out = subprocess.check_output(
            shlex.split('rocm-smi --showuse --json'), stderr=subprocess.STDOUT, text=True, timeout=0.5
        )
        short = out.strip().replace("\n", " ")
        if len(short) > 120:
            short = short[:120]
        _live_update(f"{msg} | rocm-smi: {short}")
    except Exception:
        _GPU_SMI_ENABLED = False
        return

# Batch size auto-scaler
def _autoscale_batch_size(desired_bs: int, model: nn.Module, dataset: Dataset, device, max_try: int = 3) -> int:
    candidates = []
    # Try up-scaling first
    for m in [4, 2, 1]:
        cand = desired_bs * m
        if cand not in candidates:
            candidates.append(cand)
    # And a fallback ladder
    for cand in [1024, 768, 512, 384, 256, 192, 128, 96, 64, 48, 32]:
        if cand not in candidates:
            candidates.append(cand)
    x0, y0 = dataset[0]
    in_dim = x0.shape[-1]
    for bs in candidates:
        try:
            x = torch.randn(bs, in_dim, device=device)
            y = torch.randn(bs, device=device)
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                out = model(x)
                loss = ((out - y) ** 2).mean()
            scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
            scaler.scale(loss).backward()
            scaler.step(torch.optim.SGD(model.parameters(), lr=1e-6))
            scaler.update()
            del x, y, out, loss
            torch.cuda.empty_cache()
            print(f"auto-batch OK -> {bs}")
            return bs
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"auto-batch OOM at {bs}, trying smaller...")
                torch.cuda.empty_cache()
                continue
            else:
                print(f"auto-batch runtime error at {bs}: {e}")
                continue
        except Exception as e:
            print(f"auto-batch failed at {bs}: {e}")
            continue
    print("auto-batch: fallback to desired", desired_bs)
    return desired_bs

# Initialize unified terminal+file logging (tee) early
import os as _os
_os.makedirs('inference-ready', exist_ok=True)
class _Tee(io.TextIOBase):
    def __init__(self, stream, logfile_path):
        self.stream = stream
        self.log = open(logfile_path, 'w', buffering=1)
    def write(self, s):
        self.stream.write(s)
        self.log.write(s)
        return len(s)
    def flush(self):
        self.stream.flush()
        self.log.flush()

if not isinstance(sys.stdout, _Tee):
    sys.stdout = _Tee(sys.stdout, 'inference-ready/train.log')
    sys.stderr = _Tee(sys.stderr, 'inference-ready/train.log')

# Versioning utilities

def _compute_code_version(files=None) -> str:
    if files is None:
        files = ['gpu_optuna_trainer.py']
    hasher = hashlib.sha256()
    for fp in files:
        try:
            with open(fp, 'rb') as f:
                hasher.update(f.read())
        except Exception:
            continue
    return hasher.hexdigest()

MODEL_VERSION = _compute_code_version()

def _write_code_snapshot(base_dir: str):
    code_dir = Path(base_dir) / 'model'
    code_dir.mkdir(parents=True, exist_ok=True)
    # Copy trainer file
    shutil.copy2('gpu_optuna_trainer.py', code_dir / 'gpu_optuna_trainer.py')
    # Write a minimal model_def.py with current ResNet definitions for portability
    model_def_path = code_dir / 'model_def.py'
    model_def_path.write_text(
        """
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, residual_init: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.ones(1) * float(residual_init))
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = torch.relu(self.linear1(x))
        x = self.dropout1(x)
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + self.alpha * x

class CustomResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, dropout, residual_init: float = 1.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout, residual_init=residual_init) for _ in range(num_blocks)
        ])
        self.output = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)
""".strip()
    )
    # Write version.json
    version_info = {
        'version': MODEL_VERSION,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'files': ['gpu_optuna_trainer.py', 'model/model_def.py'],
    }
    (Path(base_dir) / 'version.json').write_text(json.dumps(version_info, indent=2))
    print(f"Wrote code snapshot -> {code_dir}")
    print(f"Wrote version.json (version={MODEL_VERSION})")

def _ensure_code_snapshot(base_dir: str):
    version_file = Path(base_dir) / 'version.json'
    current_version = MODEL_VERSION
    if version_file.exists():
        try:
            stored = json.loads(version_file.read_text())
            stored_version = stored.get('version')
        except Exception:
            stored_version = None
        if stored_version != current_version:
            print(f"Version mismatch or missing (stored={stored_version}, current={current_version}) -> updating snapshot")
            _write_code_snapshot(base_dir)
        else:
            print(f"Version matches existing snapshot (version={current_version})")
    else:
        print("No version.json found -> creating snapshot")
        _write_code_snapshot(base_dir)

# GPU-ONLY ENFORCEMENT
if not torch.cuda.is_available():
    raise RuntimeError("GPU NOT AVAILABLE - REQUIRES GPU FOR TRAINING")
device = torch.device('cuda:0')
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

torch.set_float32_matmul_precision('high')

class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, residual_init: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.ones(1) * float(residual_init))
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = torch.relu(self.linear1(x))
        x = self.dropout1(x)
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + self.alpha * x

class CustomResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, dropout, residual_init: float = 1.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout, residual_init=residual_init) for _ in range(num_blocks)
        ])
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)

class TemperatureDataset(Dataset):
    def __init__(self, X, y):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def _parse_vna_timestamp_from_filename(filepath: str) -> pd.Timestamp:
    """Parse timestamp from filename like VNA-D4250801_011422.csv -> 2025-08-01 01:14:22"""
    name = Path(filepath).stem  # VNA-D4250801_011422
    try:
        base = name.split('VNA-D4')[-1]
        date_part, time_part = base.split('_')  # 250801, 011422
        year = int('20' + date_part[0:2])
        month = int(date_part[2:4])
        day = int(date_part[4:6])
        hour = int(time_part[0:2])
        minute = int(time_part[2:4])
        second = int(time_part[4:6])
        return pd.Timestamp(datetime(year, month, day, hour, minute, second))
    except Exception:
        return pd.NaT

def _print_header(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

def _print_kv(key: str, value):
    print(f"- {key}: {value}")

# User-configurable data source and feature columns (integrity-first)
# Edit these to choose which columns are used without any aliasing.
VNA_DIR = 'VNA-D4'
TEMP_CSV = 'temp_readings-D4.csv'
FEATURE_COLUMNS = [
    'Phase(deg)',
    'Rs',
]
HARD_FAIL_ON_MISSING_COLUMNS = True

def _collect_vna_raw_arrays(vna_dir: str, required_cols=None):
    """Collect raw arrays for required VNA columns per file.
    Returns list of dicts: { 'timestamp': ts, 'file': path, 'arrays': {col: np.ndarray} }
    """
    if required_cols is None:
        required_cols = FEATURE_COLUMNS
    _print_header(f"Scanning VNA directory: {vna_dir}")
    files = sorted(glob.glob(str(Path(vna_dir) / '*.csv')))
    _print_kv("files_found", len(files))
    records = []
    kept = 0
    skipped = 0
    nan_totals = {c: 0 for c in required_cols}
    total_rows = 0
    for fp in tqdm(files, desc="VNA CSVs", unit="file", file=sys.stdout):
        try:
            df = pd.read_csv(fp)
        except Exception:
            skipped += 1
            continue
        df_len = len(df)
        total_rows += df_len
        cols = {}
        ok = True
        for c in required_cols:
            col_name = c
            if c not in df.columns:
                if HARD_FAIL_ON_MISSING_COLUMNS:
                    print(f"ERROR: required column '{c}' not found in {fp}. Available: {list(df.columns)}")
                    ok = False
                    break
                else:
                    ok = False
                    break
            raw = pd.to_numeric(df[col_name], errors='coerce')
            nan_count = int(raw.isna().sum())
            nan_totals[c] += nan_count
            arr = raw.dropna().to_numpy()
            if arr.size == 0:
                ok = False
                break
            cols[c] = arr.astype(np.float32)
        if not ok:
            skipped += 1
            continue
        ts = _parse_vna_timestamp_from_filename(fp)
        if pd.isna(ts):
            skipped += 1
            continue
        records.append({
            'timestamp': ts,
            'file': fp,
            'arrays': cols,
            'lengths': {k: int(v.shape[0]) for k, v in cols.items()},
        })
        kept += 1
    _print_kv("files_kept", kept)
    _print_kv("files_skipped", skipped)
    _print_kv("csv_processed_total", kept + skipped)
    # Print NaN totals once
    print("NaN totals across processed CSVs (dropped during load):")
    for c in required_cols:
        print(f"  {c}: {nan_totals[c]}")
    if not records:
        raise RuntimeError(f"No usable VNA files in {vna_dir}")
    return records

def _determine_target_len(records, required_cols=None) -> int:
    """Determine a common per-column length to ensure equal-sized feature vectors."""
    if required_cols is None:
        required_cols = FEATURE_COLUMNS
    per_file_min = []
    for rec in records:
        min_len = min(rec['lengths'][c] for c in required_cols)
        per_file_min.append(min_len)
    median_len = int(np.median(per_file_min))
    min_len = int(min(per_file_min))
    max_len = int(max(per_file_min))
    _print_header("Determining target length")
    _print_kv("min_per_file_min", min_len)
    _print_kv("median_per_file_min", median_len)
    _print_kv("max_per_file_min", max_len)
    target_len = median_len if median_len >= 1000 else min_len
    _print_kv("target_len", target_len)
    return target_len

def _build_dense_features(records, target_len, required_cols=None):
    """Build dense, concatenated feature vectors per file."""
    if required_cols is None:
        required_cols = FEATURE_COLUMNS
    _print_header("Building dense feature vectors")
    _print_kv("required_cols", list(required_cols))
    _print_kv("target_len", target_len)
    timestamps = []
    vectors = []
    kept = 0
    short_skipped = 0
    for rec in records:
        if any(rec['lengths'][c] < target_len for c in required_cols):
            short_skipped += 1
            continue
        parts = []
        for c in required_cols:
            arr = rec['arrays'][c][:target_len]
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(arr)
        vec = np.concatenate(parts, axis=0).astype(np.float32)
        timestamps.append(rec['timestamp'])
        vectors.append(vec)
        kept += 1
    _print_kv("files_used_for_dense", kept)
    _print_kv("files_skipped_short", short_skipped)
    if not vectors:
        raise RuntimeError("All VNA files were shorter than target length; cannot build dense features")
    X = np.stack(vectors, axis=0)
    ts_series = pd.Series(timestamps)
    _print_kv("dense_X_shape", X.shape)
    return ts_series, X

# (No D5 temp copy in integrity mode)


def load_data():
    """Build dataset by aligning dense VNA vectors to nearest temperature readings (dataset-aware)."""
    _print_header("Loading data")
    print(f"Temp CSV: {TEMP_CSV}")
    print(f"VNA DIR: {VNA_DIR}")
    print(f"Feature columns: {FEATURE_COLUMNS}")
    temp_df = pd.read_csv(TEMP_CSV)
    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
    temp_df = temp_df.sort_values('timestamp').reset_index(drop=True)
    _print_kv("temp_rows", len(temp_df))

    records = _collect_vna_raw_arrays(VNA_DIR, required_cols=FEATURE_COLUMNS)
    target_len = _determine_target_len(records, required_cols=FEATURE_COLUMNS)
    vna_timestamps, X_dense = _build_dense_features(records, target_len, required_cols=FEATURE_COLUMNS)

    print("Aligning VNA to temperature readings (tolerance=15 min)")
    vna_df = pd.DataFrame({'timestamp': vna_timestamps})
    joined = pd.merge_asof(
        vna_df.sort_values('timestamp'),
        temp_df[['timestamp', 'temp_c']].sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(minutes=15),
    )
    mask = joined['temp_c'].notna().to_numpy()
    kept = int(mask.sum())
    skipped = int((~mask).sum())
    _print_kv("aligned_rows", kept)
    _print_kv("unaligned_skipped", skipped)

    X = X_dense[mask]
    y = joined.loc[mask, 'temp_c'].to_numpy(dtype=float)
    _print_kv("final_X_shape", X.shape)
    _print_kv("final_y_len", y.shape[0])
    return X, y

def split_train_val_holdout(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """70% train, 15% val (Optuna/CV), 15% holdout."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=seed)
    X_val, X_holdout, y_val, y_holdout = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed)
    _print_header("Dataset split")
    _print_kv("train_shape", X_train.shape)
    _print_kv("val_shape", X_val.shape)
    _print_kv("holdout_shape", X_holdout.shape)
    return X_train, X_val, X_holdout, y_train, y_val, y_holdout

def build_preprocess_pipeline(trial, num_features):
    """Build a sklearn Pipeline with optional steps controlled by Optuna."""
    steps = []
    use_variance_threshold = trial.suggest_categorical('use_variance_threshold', [True, False])
    if use_variance_threshold:
        threshold = trial.suggest_float('var_threshold', 0.0, 0.2)
        steps.append(('var', VarianceThreshold(threshold=threshold)))
    use_standard_scaler = trial.suggest_categorical('use_standard_scaler', [True, False])
    if use_standard_scaler:
        steps.append(('scaler', StandardScaler()))
    use_select_kbest = trial.suggest_categorical('use_select_kbest', [True, False])
    if use_select_kbest:
        k_prop = trial.suggest_int('kbest_k', 1, max(1, int(num_features)))
        steps.append(('kbest', SelectKBest(score_func=f_regression, k=k_prop)))
    print("Preprocessing steps:", [name for name, _ in steps])
    return Pipeline(steps) if steps else None

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function"""
    # Hyperparameters
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512, step=64)
    num_blocks = trial.suggest_int('num_blocks', 2, 8)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    residual_init = trial.suggest_float('residual_init', 0.0, 2.0)

    _print_header("Starting trial")
    _print_kv("hidden_dim", hidden_dim)
    _print_kv("num_blocks", num_blocks)
    _print_kv("dropout", dropout)
    _print_kv("lr", lr)
    _print_kv("batch_size", batch_size)
    _print_kv("residual_init", residual_init)
    
    X_train_raw, X_test_raw, y_train, y_test = X_train, X_val, y_train, y_val
    _print_kv("X_train_raw_shape", X_train_raw.shape)
    _print_kv("X_val_raw_shape", X_test_raw.shape)

    # Preprocess pipeline
    preprocess = build_preprocess_pipeline(trial, num_features=X_train_raw.shape[1])
    if preprocess is not None:
        X_train = preprocess.fit_transform(X_train_raw, y_train)
        X_test = preprocess.transform(X_test_raw)
        _print_kv("X_train_shape", X_train.shape)
        _print_kv("X_test_shape", X_test.shape)
    else:
        X_train, X_test = X_train_raw, X_test_raw
        _print_kv("X_train_shape", X_train.shape)
        _print_kv("X_test_shape", X_test.shape)

    input_dim = X_train.shape[1]
    _print_kv("input_dim", input_dim)

    # Create datasets and loaders
    train_dataset = TemperatureDataset(X_train, y_train)
    test_dataset = TemperatureDataset(X_test, y_test)
    
    use_cached = False
    try:
        est_bytes = X_train.nbytes + y_train.nbytes + X_test.nbytes + y_test.nbytes
        # keep headroom 2 GiB
        if est_bytes < (12 * 1024**3 - 2 * 1024**3):
            use_cached = CACHE_ON_GPU
    except Exception:
        use_cached = False
    _print_kv("gpu_cache_enabled", use_cached)

    if use_cached:
        X_train_t, y_train_t = _cache_arrays_to_gpu(X_train, y_train, device)
        X_test_t, y_test_t = _cache_arrays_to_gpu(X_test, y_test, device)
        effective_bs = max(256, batch_size)
        _print_kv("effective_batch_size", effective_bs)
    else:
        # Autoscale batch size and use DataLoaders
        temp_model = CustomResNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            residual_init=residual_init,
        ).to(device)
        if USE_COMPILE:
            try:
                temp_model = torch.compile(temp_model)
            except Exception:
                pass
        effective_bs = _autoscale_batch_size(batch_size, temp_model, train_dataset, device)
        del temp_model
        torch.cuda.empty_cache()
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_bs,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=effective_bs,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        _print_kv("effective_batch_size", effective_bs)

    model = CustomResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        residual_init=residual_init,
    ).to(device)
    if USE_COMPILE:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile not used: {e}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_r2 = -float('inf')
    patience = 20
    patience_counter = 0
    min_epochs = 200

    with Live(refresh_per_second=4, transient=False) as live:
        for epoch in range(1000):
            if use_cached:
                avg_train_loss, r2 = _train_eval_epoch_cached(
                    model, X_train_t, y_train_t, X_test_t, y_test_t, effective_bs, criterion, optimizer, scaler, epoch, USE_AMP
                )
            else:
                model.train()
                train_loss_total = 0.0
                num_batches = 0
                # manual loop without tqdm to keep single-line live display
                for batch_start in range(0, len(train_dataset), effective_bs):
                    batch_X = torch.as_tensor(X_train[batch_start:batch_start+effective_bs], device=device)
                    batch_y = torch.as_tensor(y_train[batch_start:batch_start+effective_bs], device=device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss_total += float(loss.item())
                    num_batches += 1
                avg_train_loss = train_loss_total / max(1, num_batches)

                model.eval()
                all_preds = []
                all_targets = []
                with torch.no_grad():
                    for batch_start in range(0, len(test_dataset), effective_bs):
                        batch_X = torch.as_tensor(X_test[batch_start:batch_start+effective_bs], device=device)
                        batch_y = torch.as_tensor(y_test[batch_start:batch_start+effective_bs], device=device)
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                            outputs = model(batch_X)
                        all_preds.extend(outputs.float().cpu().numpy())
                        all_targets.extend(batch_y.float().cpu().numpy())
                preds_t = torch.tensor(np.array(all_preds), dtype=torch.float32, device=device)
                targets_t = torch.tensor(np.array(all_targets), dtype=torch.float32, device=device)
                r2_skl, _ = r2_both(targets_t, preds_t)
                r2 = r2_skl

            live.update(f"Trial running | epoch={epoch:04d} | train_loss={avg_train_loss:.6f} | val_R2={r2:.6f} | bs={effective_bs}")
            if r2 > best_r2:
                best_r2 = r2
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience and (epoch + 1) >= min_epochs:
                print("\nEarly stop on patience after min_epochs")
                break
            trial.report(r2, epoch)
            if trial.should_prune():
                print("\nTrial pruned")
                raise optuna.TrialPruned()
    _live_newline()
    return best_r2

def save_inference_artifacts(model, best_params, r2_value, preprocess, X_holdout_raw, y_holdout, X_all, y_all):
    """Save model, params, fitted preprocessing pipeline, and holdout features for inference."""
    _print_header("Saving inference artifacts")
    # Create inference-ready directory
    import os
    os.makedirs('inference-ready', exist_ok=True)

    # Ensure code snapshot/versioning
    _ensure_code_snapshot('inference-ready')
    
    # Save model
    torch.save(model.state_dict(), 'inference-ready/resnet_model.pth')
    print("Saved model weights -> inference-ready/resnet_model.pth")
    
    # Save parameters (including preprocessing toggles)
    with open('inference-ready/model_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print("Saved model params -> inference-ready/model_params.json")

    # Save metrics
    with open('inference-ready/metrics.json', 'w') as f:
        json.dump({'r2_score': r2_value, 'version': MODEL_VERSION}, f, indent=2)
    print("Saved metrics -> inference-ready/metrics.json")

    # Save individual preprocessing components if they exist
    if preprocess is not None:
        for step_name, step_transformer in preprocess.steps:
            if step_name == 'scaler':
                joblib.dump(step_transformer, 'inference-ready/scaler.pkl')
                print("Saved StandardScaler -> inference-ready/scaler.pkl")
            elif step_name == 'var':
                joblib.dump(step_transformer, 'inference-ready/var_thresh.pkl')
                print("Saved VarianceThreshold -> inference-ready/var_thresh.pkl")
            elif step_name == 'kbest':
                joblib.dump(step_transformer, 'inference-ready/kbest_selector.pkl')
                print("Saved SelectKBest -> inference-ready/kbest_selector.pkl")
    
    # Save holdout features and targets for true inference-time evaluation
    np.save('inference-ready/X_holdout_raw.npy', X_holdout_raw)
    np.save('inference-ready/y_holdout.npy', y_holdout)
    print("Saved holdout arrays -> inference-ready/X_holdout_raw.npy, y_holdout.npy")
    
    # Save data statistics for reference (use cached arrays, do not reload)
    data_stats = {
        'X_shape': list(X_all.shape),
        'y_shape': [int(y_all.shape[0])],
        'X_min': X_all.min(axis=0).tolist(),
        'X_max': X_all.max(axis=0).tolist(),
        'y_mean': float(y_all.mean()),
        'y_std': float(y_all.std()),
        'r2_score': r2_value,
        'preprocessing_steps': [step[0] for step in preprocess.steps] if preprocess is not None else [],
        'version': MODEL_VERSION,
    }
    with open('inference-ready/data_stats.json', 'w') as f:
        json.dump(data_stats, f, indent=2)
    print("Saved data stats -> inference-ready/data_stats.json")
    print(f"Model saved with holdout R² = {r2_value:.6f}")

def train_final_model(best_params, X_trainval_raw, y_trainval, X_holdout_raw, y_holdout):
    """Train final model with best parameters on train+val, evaluate on holdout."""
    _print_header("Final training with best parameters")
    for k, v in best_params.items():
        _print_kv(k, v)
    
    # Use provided train+val as training base
    X_train_raw = X_trainval_raw
    y_train = y_trainval
    _print_kv("X_trainval_raw_shape", X_train_raw.shape)
    _print_kv("X_holdout_raw_shape", X_holdout_raw.shape)

    # Recreate preprocessing pipeline from best_params
    class BestParamsAccessor:
        def __init__(self, params):
            self.params = params
        def suggest_categorical(self, name, choices):
            return self.params.get(name, False)
        def suggest_float(self, name, low, high):
            return self.params.get(name, low)
        def suggest_int(self, name, low, high, step=None):
            return self.params.get(name, low)
    accessor = BestParamsAccessor(best_params)

    preprocess = build_preprocess_pipeline(accessor, num_features=X_train_raw.shape[1])

    if preprocess is not None:
        X_train = preprocess.fit_transform(X_train_raw, y_train)
        X_holdout = preprocess.transform(X_holdout_raw)
    else:
        X_train, X_holdout = X_train_raw, X_holdout_raw
    _print_kv("X_train_shape", X_train.shape)
    _print_kv("X_holdout_shape", X_holdout.shape)

    input_dim = X_train.shape[1]
    _print_kv("input_dim", input_dim)

    # Create datasets and loaders (monitor with a small internal split from train)
    # Keep 10% of training as internal eval for live metrics/early stopping
    n_train = X_train.shape[0]
    split_idx = max(1, int(0.9 * n_train))
    X_tr_int, X_ev_int = X_train[:split_idx], X_train[split_idx:]
    y_tr_int, y_ev_int = y_train[:split_idx], y_train[split_idx:]
    train_dataset = TemperatureDataset(X_tr_int, y_tr_int)
    test_dataset = TemperatureDataset(X_ev_int, y_ev_int)
    
    use_cached = False
    try:
        est_bytes = X_train.nbytes + y_train.nbytes + X_ev_int.nbytes + y_ev_int.nbytes
        if est_bytes < (12 * 1024**3 - 2 * 1024**3):
            use_cached = CACHE_ON_GPU
    except Exception:
        use_cached = False
    _print_kv("gpu_cache_enabled", use_cached)

    if use_cached:
        X_train_t, y_train_t = _cache_arrays_to_gpu(X_train, y_train, device)
        X_test_t, y_test_t = _cache_arrays_to_gpu(X_ev_int, y_ev_int, device)
        effective_bs = max(256, best_params['batch_size'])
        _print_kv("effective_batch_size", effective_bs)
    else:
        effective_bs = _autoscale_batch_size(best_params['batch_size'], CustomResNet(
            input_dim=input_dim,
            hidden_dim=best_params['hidden_dim'],
            num_blocks=best_params['num_blocks'],
            dropout=best_params['dropout'],
            residual_init=best_params.get('residual_init', 1.0),
        ).to(device), train_dataset, device)
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_bs,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=effective_bs,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        _print_kv("effective_batch_size", effective_bs)

    model = CustomResNet(
        input_dim=input_dim,
        hidden_dim=best_params['hidden_dim'],
        num_blocks=best_params['num_blocks'],
        dropout=best_params['dropout'],
        residual_init=best_params.get('residual_init', 1.0),
    ).to(device)
    if USE_COMPILE:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile not used: {e}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    # Training loop
    best_r2 = -float('inf')
    best_model_state = None
    last_preds, last_targets = None, None
    min_epochs = 200
    patience = 20
    patience_counter = 0
    
    with Live(refresh_per_second=4, transient=False) as live:
        for epoch in range(2000):
            if use_cached:
                avg_train_loss, r2 = _train_eval_epoch_cached(
                    model, X_train_t, y_train_t, X_test_t, y_test_t, effective_bs, criterion, optimizer, scaler, epoch, USE_AMP
                )
            else:
                model.train()
                train_loss_total = 0.0
                num_batches = 0
                for batch_start in range(0, len(train_dataset), effective_bs):
                    batch_X = torch.as_tensor(X_train[batch_start:batch_start+effective_bs], device=device)
                    batch_y = torch.as_tensor(y_train[batch_start:batch_start+effective_bs], device=device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss_total += float(loss.item())
                    num_batches += 1
                avg_train_loss = train_loss_total / max(1, num_batches)

                model.eval()
                all_preds = []
                all_targets = []
                with torch.no_grad():
                    for batch_start in range(0, len(test_dataset), effective_bs):
                        batch_X = torch.as_tensor(X_ev_int[batch_start:batch_start+effective_bs], device=device)
                        batch_y = torch.as_tensor(y_ev_int[batch_start:batch_start+effective_bs], device=device)
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                            outputs = model(batch_X)
                        all_preds.extend(outputs.float().cpu().numpy())
                        all_targets.extend(batch_y.float().cpu().numpy())
                preds_t = torch.tensor(np.array(all_preds), dtype=torch.float32, device=device)
                targets_t = torch.tensor(np.array(all_targets), dtype=torch.float32, device=device)
                r2_skl, _ = r2_both(targets_t, preds_t)
                r2 = r2_skl

            _live_update(f"Final train | epoch={epoch:04d} | train_loss={avg_train_loss:.6f} | int_val_R2={r2:.6f} | bs={effective_bs}")
            if r2 > best_r2:
                best_r2 = r2
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            if (epoch + 1) >= min_epochs and r2 > 0.98:
                print(f"\nTarget R² > 0.98 achieved: {r2:.6f}")
                break
            if (epoch + 1) >= min_epochs and patience_counter >= patience:
                print("\nEarly stop on patience after min_epochs")
                break
    _live_newline()
    
    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluate on true holdout (once, no training impact)
    model.eval()
    with torch.no_grad():
        Xh_t = torch.as_tensor(X_holdout, dtype=torch.float32, device=device)
        yh_t = torch.as_tensor(y_holdout, dtype=torch.float32, device=device)
        preds_h = model(Xh_t)
    r2_holdout, _ = r2_both(yh_t, preds_h)
    _print_kv("holdout_R2", r2_holdout)

    return model, r2_holdout, (last_preds, last_targets), preprocess, X_holdout_raw, y_holdout

def main():
    """Main training pipeline"""
    print("Starting GPU-Only Optuna Training...")
    print(f"Target: R² > 0.98")
    
    # Cache dataset once to avoid re-processing per trial
    X, y = load_data()
    X_train, X_val, X_holdout, y_train, y_val, y_holdout = split_train_val_holdout(X, y)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize with cached X, y
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=120,
        timeout=10800,
        show_progress_bar=True,
    )

    # Save trials history
    try:
        df_trials = study.trials_dataframe()
        df_trials.to_csv('inference-ready/optuna_trials.csv', index=False)
        print("Saved Optuna trials history -> inference-ready/optuna_trials.csv")
    except Exception as e:
        print(f"Failed to save Optuna trials history: {e}")
    
    print(f"Best trial R²: {study.best_value:.6f}")
    print(f"Best parameters: {study.best_params}")
    
    # Train final model with cached arrays
    # Train final model on train+val, test on holdout
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    model, final_r2, (preds, targets), preprocess, X_holdout_raw, y_holdout_arr = train_final_model(study.best_params, X_trainval, y_trainval, X_holdout, y_holdout)

    # Save artifacts using cached arrays for stats and holdout
    save_inference_artifacts(model, study.best_params, final_r2, preprocess, X_holdout_raw, y_holdout_arr, X, y)
    
    print("Training complete!")
    print(f"Final R²: {final_r2:.6f}")

if __name__ == "__main__":
    main()
