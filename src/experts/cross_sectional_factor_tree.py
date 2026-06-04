#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V2H Factor Tree Expert.

Trains a cross-sectional tabular factor expert using daily stock/cross features.
Preferred: LightGBM if installed.
Fallback: sklearn HistGradientBoostingRegressor.

Output:
  score_val_factor_tree.csv
  score_test_factor_tree.csv
  final_factor_tree_report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_memmap(path, shape, dtype="float32"):
    return np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape))


def normalize_meta(meta):
    meta = meta.copy()
    ren = {}
    for c in meta.columns:
        lc = c.lower()
        if lc == "date":
            ren[c] = "Date"
        elif lc in ["ticker", "ts_code", "code", "stock_code"]:
            ren[c] = "Ticker"
    meta = meta.rename(columns=ren)
    meta["Date"] = pd.to_datetime(meta["Date"]).dt.strftime("%Y-%m-%d")
    meta["Ticker"] = meta["Ticker"].astype(str)
    return meta


def choose_split(meta, split_name):
    """
    Robust split resolver.

    Priority:
    1) explicit split columns if available
    2) source_split if available
    3) fixed chronological DIVE/MSAN split by Date
    """
    for col in ["split_name", "period", "source_split", "split"]:
        if col in meta.columns:
            m = meta[col].astype(str).str.lower()
            hit = meta[m == split_name].index.values
            if len(hit) > 0:
                return hit

    d = pd.to_datetime(meta["Date"])

    if split_name == "train":
        mask = (d >= "2016-01-04") & (d <= "2021-12-31")
    elif split_name in ["val", "valid", "validation"]:
        mask = (d >= "2022-01-04") & (d <= "2023-12-29")
    elif split_name == "test":
        mask = (d >= "2024-01-02") & (d <= "2026-04-23")
    else:
        raise ValueError(f"Unknown split_name={split_name}")

    idx = meta[mask].index.values
    if len(idx) == 0:
        raise ValueError(f"No rows found for split={split_name}; Date range fallback failed.")
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_external/msan_samples_full_qfq_norm")
    ap.add_argument("--out_dir", default="outputs/dive_trader_v2/experts/factor_tree")
    ap.add_argument("--target_col", default="future_return_5d")
    ap.add_argument("--max_train_rows", type=int, default=800000)
    ap.add_argument("--max_val_rows", type=int, default=300000)
    ap.add_argument("--random_seed", type=int, default=42)
    args = ap.parse_args()

    print("[BOOT] start factor_tree expert", flush=True)
    rng = np.random.default_rng(args.random_seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[LOAD] feature_info/meta", flush=True)
    feature_info = read_json(data_dir / "feature_info_L60_H5_full.json")
    meta = normalize_meta(pd.read_parquet(data_dir / "meta_L60_H5_full.parquet"))

    n = len(meta)

    x_cross_shape = feature_info.get("X_cross_today_shape") or [n, 13]
    x_stock_shape = feature_info.get("X_stock_seq_shape") or [n, 60, 26]

    X_cross = load_memmap(data_dir / "X_cross_today_L60_H5_full.dat", x_cross_shape)
    X_stock = load_memmap(data_dir / "X_stock_seq_L60_H5_full.dat", x_stock_shape)

    y_path = data_dir / "y_ret_5d_L60_H5_full.dat"
    y = np.memmap(y_path, dtype="float32", mode="r", shape=(n,))
    # Canonical return target is stored in percentage points.
    # Convert to decimal return for model training/backtest consistency.
    y = np.asarray(y, dtype="float32") / 100.0

    train_idx = choose_split(meta, "train")
    val_idx = choose_split(meta, "val")
    test_idx = choose_split(meta, "test")

    if len(train_idx) > args.max_train_rows:
        train_idx = rng.choice(train_idx, size=args.max_train_rows, replace=False)
    if len(val_idx) > args.max_val_rows:
        val_fit_idx = rng.choice(val_idx, size=args.max_val_rows, replace=False)
    else:
        val_fit_idx = val_idx

    def build_features(idx):
        # Cross ranks + latest stock features + simple temporal summaries.
        xs_last = np.asarray(X_stock[idx, -1, :], dtype=np.float32)
        xs_mean20 = np.asarray(X_stock[idx, -20:, :].mean(axis=1), dtype=np.float32)
        xs_std20 = np.asarray(X_stock[idx, -20:, :].std(axis=1), dtype=np.float32)
        xc = np.asarray(X_cross[idx], dtype=np.float32)
        return np.concatenate([xc, xs_last, xs_mean20, xs_std20], axis=1)

    print(f"[BUILD] X_train rows={len(train_idx)}", flush=True)
    X_tr = build_features(train_idx)
    y_tr = np.asarray(y[train_idx], dtype=np.float32)
    print(f"[BUILD] X_val_fit rows={len(val_fit_idx)}", flush=True)
    X_va_fit = build_features(val_fit_idx)
    y_va_fit = np.asarray(y[val_fit_idx], dtype=np.float32)

    if HAS_LGB:
        model_type = "LightGBMRegressor"
        model = lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=args.random_seed,
            n_jobs=-1,
            device_type="gpu",
            gpu_platform_id=0,
            gpu_device_id=0,
            max_bin=63,
            verbosity=1,
        )
        print("[FIT] LightGBM start", flush=True)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va_fit, y_va_fit)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        model_type = "HistGradientBoostingRegressor"
        model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.04,
            max_leaf_nodes=63,
            l2_regularization=0.1,
            random_state=args.random_seed,
        )
        model.fit(X_tr, y_tr)

    def predict_split(idx, name):
        X = build_features(idx)
        pred = model.predict(X)
        out = meta.iloc[idx][["Date", "Ticker"]].copy()
        out["score"] = pred.astype(float)
        path = out_dir / f"score_{name}_factor_tree.csv"
        out.to_csv(path, index=False)
        return path, pred

    print("[PRED] validation/test full split", flush=True)
    val_path, pred_val = predict_split(val_idx, "val")
    test_path, pred_test = predict_split(test_idx, "test")

    report = {
        "method": "V2H Factor Tree Expert",
        "model_type": model_type,
        "data_dir": str(data_dir),
        "n_train_fit": int(len(train_idx)),
        "n_val_fit": int(len(val_fit_idx)),
        "n_val_total": int(len(val_idx)),
        "n_test_total": int(len(test_idx)),
        "feature_dim": int(X_tr.shape[1]),
        "val_pred_mean": float(np.mean(pred_val)),
        "test_pred_mean": float(np.mean(pred_test)),
        "train_mse_sample": float(mean_squared_error(y_tr[:min(len(y_tr), 50000)], model.predict(X_tr[:min(len(y_tr), 50000)]))),
        "outputs": [val_path.name, test_path.name, "final_factor_tree_report.json"],
    }
    (out_dir / "final_factor_tree_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
