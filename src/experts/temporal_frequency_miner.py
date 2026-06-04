#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def split_idx(meta, split):
    d = pd.to_datetime(meta["Date"])
    if split == "train":
        m = (d >= "2016-01-04") & (d <= "2021-12-31")
    elif split == "val":
        m = (d >= "2022-01-04") & (d <= "2023-12-29")
    elif split == "test":
        m = (d >= "2024-01-02") & (d <= "2026-04-23")
    else:
        raise ValueError(split)
    return meta.index[m].to_numpy()


def max_weight_projection(raw, exposure, max_weight):
    raw = np.maximum(np.asarray(raw, float), 0.0)
    if raw.sum() <= 1e-12:
        w = np.ones_like(raw) / len(raw) * exposure
    else:
        w = raw / raw.sum() * exposure
    for _ in range(30):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if under.sum() == 0 or excess <= 1e-12:
            break
        if w[under].sum() <= 1e-12:
            w[under] += excess / under.sum()
        else:
            w[under] += excess * w[under] / w[under].sum()
    return w


def summarize(daily, strategy, horizon_days):
    ret = daily["ret"].to_numpy(float)
    all_ret = daily["all_ret"].to_numpy(float)
    mean = float(np.mean(ret))
    std = float(np.std(ret, ddof=0))
    sharpe = mean / std if std > 1e-12 else 0.0
    eq = np.cumprod(1.0 + ret)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    out = {
        "strategy": strategy,
        "n_days": int(len(daily)),
        "mean_ret": mean,
        "std_ret": std,
        "sharpe_like": sharpe,
        "max_drawdown_like": float(dd.min()),
        "positive_day_ratio": float((ret > 0).mean()),
        "mean_all_ret": float(np.mean(all_ret)),
        "mean_excess": float(np.mean(ret - all_ret)),
        "turnover_mean": float(daily["turnover"].mean()),
        "n_holding_mean": float(daily["n_holding"].mean()),
        "max_weight_mean": float(daily["max_weight"].mean()),
        "top3_weight_mean": float(daily["top3_weight"].mean()),
        "top5_weight_mean": float(daily["top5_weight"].mean()),
        "herfindahl_mean": float(daily["herfindahl"].mean()),
        "ann_ret_like": mean * 252.0 / horizon_days,
        "ann_sharpe_like": sharpe * np.sqrt(252.0 / horizon_days),
    }
    out["objective"] = (
        out["sharpe_like"]
        + 0.5 * out["mean_excess"]
        + 0.2 * out["positive_day_ratio"]
        + 0.6 * out["max_drawdown_like"]
        - 0.05 * out["turnover_mean"]
        - 0.10 * out["top3_weight_mean"]
    )
    return out


def build_freq_features(idx, X_stock, X_cross):
    idx = np.sort(idx)
    seq = np.asarray(X_stock[idx], dtype=np.float32)  # [N,60,F]
    xc = np.asarray(X_cross[idx], dtype=np.float32)

    # Use all 26 channels, compact spectral statistics.
    x = seq - seq.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(x, axis=1)
    amp = np.abs(spec).astype(np.float32)  # [N,31,F]

    low = amp[:, 1:4, :].mean(axis=1)
    mid = amp[:, 4:10, :].mean(axis=1)
    high = amp[:, 10:, :].mean(axis=1)
    total = amp[:, 1:, :].sum(axis=1) + 1e-6

    low_ratio = low / total
    high_ratio = high / total
    spectral_entropy = -(amp[:, 1:, :] / total[:, None, :] * np.log((amp[:, 1:, :] / total[:, None, :]) + 1e-8)).sum(axis=1)

    last = seq[:, -1, :]
    mean20 = seq[:, -20:, :].mean(axis=1)
    std20 = seq[:, -20:, :].std(axis=1)

    return np.concatenate([xc, last, mean20, std20, low, mid, high, low_ratio, high_ratio, spectral_entropy], axis=1).astype(np.float32)


def backtest_scores(meta_split, score, y_decimal, idx, split, args):
    tab = meta_split[["Date", "Ticker"]].copy().reset_index(drop=True)
    tab["score"] = score.astype(float)
    tab["y"] = np.asarray(y_decimal[idx], dtype=np.float32)

    daily_rows, ledger_rows = [], []
    prev_w = {}

    for date, g in tab.groupby("Date", sort=True):
        g = g.sort_values("score", ascending=False).head(args.topk).copy()
        z = g["score"].to_numpy(float)
        z = z - np.nanmax(z)
        raw = np.exp(z / args.temp)
        w = max_weight_projection(raw, args.exposure, args.max_weight)

        tickers = g["Ticker"].astype(str).tolist()
        r = g["y"].to_numpy(float)
        gross = float(np.sum(w * r))

        cur_w = {t: float(x) for t, x in zip(tickers, w)}
        all_names = set(cur_w) | set(prev_w)
        turnover = sum(abs(cur_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in all_names)
        cost = turnover * args.cost_bps / 10000.0
        ret = gross - cost
        all_ret = float(tab.loc[tab["Date"] == date, "y"].mean())

        sw = np.sort(w)[::-1]
        daily_rows.append({
            "Date": date,
            "ret": ret,
            "gross_ret": gross,
            "all_ret": all_ret,
            "cost": cost,
            "turnover": turnover,
            "n_holding": len(w),
            "max_weight": float(sw[0]),
            "top3_weight": float(sw[:3].sum()),
            "top5_weight": float(sw[:5].sum()),
            "herfindahl": float(np.sum(w ** 2)),
        })

        for rank, (t, ww, rr, ss) in enumerate(zip(tickers, w, r, z), start=1):
            ledger_rows.append({
                "Date": date,
                "Ticker": t,
                "rank": rank,
                "weight": float(ww),
                "future_return_5d": float(rr),
                "score": float(ss),
                "gross_contribution": float(ww * rr),
                "day_gross_ret": gross,
                "day_cost": cost,
                "day_net_ret": ret,
                "day_all_ret": all_ret,
                "day_turnover": turnover,
            })

        prev_w = cur_w

    return pd.DataFrame(daily_rows), pd.DataFrame(ledger_rows), tab[["Date", "Ticker", "score"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_external/msan_samples_full_qfq_norm")
    ap.add_argument("--out_dir", default="outputs/dive_trader_v2/experts/v2l_frequency_expert")
    ap.add_argument("--max_train_rows", type=int, default=300000)
    ap.add_argument("--max_val_rows", type=int, default=100000)
    ap.add_argument("--horizon_days", type=float, default=5.0)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--exposure", type=float, default=0.5)
    ap.add_argument("--max_weight", type=float, default=0.10)
    ap.add_argument("--temp", type=float, default=0.35)
    ap.add_argument("--random_seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.random_seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[LOAD] meta/features")
    feature_info = read_json(data_dir / "feature_info_L60_H5_full.json")
    meta = normalize_meta(pd.read_parquet(data_dir / "meta_L60_H5_full.parquet"))
    n = len(meta)

    X_cross = load_memmap(data_dir / "X_cross_today_L60_H5_full.dat", feature_info.get("X_cross_today_shape") or [n, 13])
    X_stock = load_memmap(data_dir / "X_stock_seq_L60_H5_full.dat", feature_info.get("X_stock_seq_shape") or [n, 60, 26])
    y = np.asarray(np.memmap(data_dir / "y_ret_5d_L60_H5_full.dat", dtype="float32", mode="r", shape=(n,)), dtype=np.float32) / 100.0

    train_idx = split_idx(meta, "train")
    val_idx = split_idx(meta, "val")
    test_idx = split_idx(meta, "test")

    if len(train_idx) > args.max_train_rows:
        train_fit_idx = np.sort(rng.choice(train_idx, size=args.max_train_rows, replace=False))
    else:
        train_fit_idx = np.sort(train_idx)

    if len(val_idx) > args.max_val_rows:
        val_fit_idx = np.sort(rng.choice(val_idx, size=args.max_val_rows, replace=False))
    else:
        val_fit_idx = np.sort(val_idx)

    print(f"[BUILD] X_train rows={len(train_fit_idx)}")
    X_tr = build_freq_features(train_fit_idx, X_stock, X_cross)
    y_tr = y[train_fit_idx]

    print(f"[BUILD] X_val_fit rows={len(val_fit_idx)}")
    X_va_fit = build_freq_features(val_fit_idx, X_stock, X_cross)
    y_va_fit = y[val_fit_idx]

    print("[FIT] frequency expert")
    if HAS_LGB:
        model_type = "LightGBMRegressor_GPU"
        model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.035,
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
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va_fit, y_va_fit)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
    else:
        model_type = "HistGradientBoostingRegressor"
        model = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.04,
            max_leaf_nodes=63,
            l2_regularization=0.1,
            random_state=args.random_seed,
        )
        model.fit(X_tr, y_tr)

    def predict_in_chunks(idx, name, chunk_size=200000):
        idx = np.sort(idx)
        preds = []
        print(f"[PRED] {name} full rows={len(idx)} chunk_size={chunk_size}", flush=True)
        for st in range(0, len(idx), chunk_size):
            ed = min(st + chunk_size, len(idx))
            print(f"[PRED] {name} chunk {st}:{ed}", flush=True)
            X_chunk = build_freq_features(idx[st:ed], X_stock, X_cross)
            preds.append(model.predict(X_chunk))
        return idx, np.concatenate(preds)

    val_idx, pred_val = predict_in_chunks(val_idx, "val")
    test_idx, pred_test = predict_in_chunks(test_idx, "test")

    val_meta = meta.iloc[val_idx][["Date", "Ticker"]].copy().reset_index(drop=True)
    test_meta = meta.iloc[test_idx][["Date", "Ticker"]].copy().reset_index(drop=True)

    print("[BACKTEST] val/test")
    daily_val, ledger_val, score_val = backtest_scores(val_meta, pred_val, y, val_idx, "val", args)
    daily_test, ledger_test, score_test = backtest_scores(test_meta, pred_test, y, test_idx, "test", args)

    score_val.to_csv(out_dir / "score_val_v2l_frequency_expert.csv", index=False)
    score_test.to_csv(out_dir / "score_test_v2l_frequency_expert.csv", index=False)
    daily_val.to_csv(out_dir / "daily_val_v2l_frequency_expert.csv", index=False)
    daily_test.to_csv(out_dir / "daily_test_v2l_frequency_expert.csv", index=False)
    ledger_val.to_csv(out_dir / "ledger_val_v2l_frequency_expert.csv", index=False)
    ledger_test.to_csv(out_dir / "ledger_test_v2l_frequency_expert.csv", index=False)

    summary = pd.DataFrame([
        {**summarize(daily_val, "v2l_frequency_aware_expert", args.horizon_days), "split": "val"},
        {**summarize(daily_test, "v2l_frequency_aware_expert", args.horizon_days), "split": "test"},
    ])
    summary.to_csv(out_dir / "summary_v2l_frequency_expert.csv", index=False)

    report = {
        "method": "V2L Frequency-aware Temporal Expert",
        "formula": "FFT spectral statistics over stock sequences + cross-sectional features -> LightGBM score -> TopK portfolio.",
        "selection_protocol": "Train on train split; validate during fitting; apply frozen expert to test.",
        "model_type": model_type,
        "config": vars(args),
        "train_mse_sample": float(mean_squared_error(y_tr[:min(len(y_tr), 50000)], model.predict(X_tr[:min(len(y_tr), 50000)]))),
        "val": summary.iloc[0].to_dict(),
        "test": summary.iloc[1].to_dict(),
        "outputs": [
            "summary_v2l_frequency_expert.csv",
            "score_val_v2l_frequency_expert.csv",
            "score_test_v2l_frequency_expert.csv",
            "daily_val_v2l_frequency_expert.csv",
            "daily_test_v2l_frequency_expert.csv",
            "ledger_val_v2l_frequency_expert.csv",
            "ledger_test_v2l_frequency_expert.csv",
        ],
    }
    (out_dir / "final_v2l_frequency_expert_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== SUMMARY =====")
    print(summary.to_string(index=False))
    print(f"[OK] saved: {out_dir}")


if __name__ == "__main__":
    main()
