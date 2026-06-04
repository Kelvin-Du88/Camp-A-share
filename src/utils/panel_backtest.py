from pathlib import Path
import json
import numpy as np
import pandas as pd


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_score(path, name):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.zfill(6)
    df = df[["Date", "Ticker", "score"]].rename(columns={"score": name})
    return df


def normalize_return_unit(s):
    y = pd.to_numeric(s, errors="coerce").astype("float64")
    # dataset meta stores 5-day return in percent-like unit, e.g. 23.6 means +23.6%
    if y.abs().quantile(0.999) > 1.0:
        y = y / 100.0
    return y


def load_meta_returns(data_dir, split):
    data_dir = Path(data_dir)
    meta = pd.read_parquet(data_dir / "meta_L60_H5_full_with_period_label.parquet")
    meta["Date"] = pd.to_datetime(meta["Date"])
    meta["Ticker"] = meta["Ticker"].astype(str).str.zfill(6)

    if split == "val":
        m = meta[meta["period_label"].eq("val")].copy()
    elif split == "test":
        m = meta[meta["period_label"].astype(str).str.startswith("test")].copy()
    else:
        m = meta[meta["period_label"].eq(split)].copy()

    ret_col = "future_return_5d_clipped" if "future_return_5d_clipped" in m.columns else "future_return_5d"
    m["future_return_5d"] = normalize_return_unit(m[ret_col])
    return m[["Date", "Ticker", "future_return_5d"]]


def load_panel(data_dir, split, score_paths):
    ret = load_meta_returns(data_dir, split)
    panel = ret
    for name, path in score_paths.items():
        p = Path(path)
        if not p.exists():
            print(f"[MISS] {name}: {p}", flush=True)
            continue
        s = read_score(p, name)
        panel = panel.merge(s, on=["Date", "Ticker"], how="inner")
        print(f"[MERGE] {split} {name}: {panel.shape}", flush=True)
    panel = panel.dropna().copy()
    return panel


def zscore_by_date(df, cols):
    out = df.copy()
    for c in cols:
        g = out.groupby("Date")[c]
        mu = g.transform("mean")
        sd = g.transform("std").replace(0, np.nan)
        out[c + "_z"] = ((out[c] - mu) / (sd + 1e-9)).clip(-5, 5).fillna(0.0)
    return out


def project_topk(day, score_col, topk=30, exposure=0.5, max_weight=0.10, temp=0.10):
    d = day.dropna(subset=[score_col, "future_return_5d"]).copy()
    if d.empty:
        return pd.DataFrame()

    d = d.sort_values(score_col, ascending=False).head(topk).copy()
    if d.empty:
        return pd.DataFrame()

    x = d[score_col].astype("float64").to_numpy()
    x = x - np.nanmax(x)
    w = np.exp(x / max(temp, 1e-6))
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(len(d), dtype="float64")
    w = w / w.sum() * exposure
    w = np.minimum(w, max_weight)
    if w.sum() > 0:
        w = w / w.sum() * exposure

    d["weight"] = w
    d["rank"] = np.arange(1, len(d) + 1)
    return d[["Date", "Ticker", "rank", "weight", "future_return_5d", score_col]]


def backtest(panel, score_col, topk=30, exposure=0.5, max_weight=0.10, temp=0.10, cost_bps=5.0, gate_daily=None):
    ledgers = []
    prev = {}
    cost_rate = cost_bps / 10000.0

    for dt, day in panel.groupby("Date", sort=True):
        hold = project_topk(day, score_col, topk, exposure, max_weight, temp)
        if hold.empty:
            continue

        gate = 1.0
        if gate_daily is not None:
            gate = float(gate_daily.get(pd.Timestamp(dt), 1.0))
            gate = max(0.0, min(1.0, gate))
            hold["weight"] *= gate

        cur = dict(zip(hold["Ticker"], hold["weight"]))
        keys = set(prev) | set(cur)
        turnover = sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)

        gross = float((hold["weight"] * hold["future_return_5d"]).sum())
        cost = turnover * cost_rate
        net = gross - cost
        all_ret = float(day["future_return_5d"].mean())

        hold["gross_contribution"] = hold["weight"] * hold["future_return_5d"]
        hold["day_gross_ret"] = gross
        hold["day_cost"] = cost
        hold["day_net_ret"] = net
        hold["day_all_ret"] = all_ret
        hold["day_turnover"] = turnover
        hold["gate"] = gate
        ledgers.append(hold)
        prev = cur

    if not ledgers:
        return pd.DataFrame(), pd.DataFrame(), {}

    ledger = pd.concat(ledgers, ignore_index=True)
    daily = ledger.groupby("Date", as_index=False).agg(
        ret=("day_net_ret", "first"),
        gross_ret=("day_gross_ret", "first"),
        all_ret=("day_all_ret", "first"),
        cost=("day_cost", "first"),
        turnover=("day_turnover", "first"),
        n_holding=("Ticker", "count"),
        gate=("gate", "first"),
        max_weight=("weight", "max"),
    )

    r = daily["ret"].astype(float)
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    mean = float(r.mean()) if len(r) else 0.0
    sharpe = mean / (std + 1e-12)
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    maxdd = float(dd.min()) if len(dd) else 0.0

    summary = {
        "n_days": int(len(daily)),
        "mean_ret": mean,
        "std_ret": std,
        "sharpe_like": sharpe,
        "ann_ret_like": mean * 252 / 5,
        "ann_sharpe_like": sharpe * np.sqrt(252 / 5),
        "max_drawdown_like": maxdd,
        "positive_day_ratio": float((r > 0).mean()) if len(r) else 0.0,
        "mean_all_ret": float(daily["all_ret"].mean()) if len(daily) else 0.0,
        "mean_excess": mean - float(daily["all_ret"].mean()) if len(daily) else 0.0,
        "turnover_mean": float(daily["turnover"].mean()) if len(daily) else 0.0,
        "n_holding_mean": float(daily["n_holding"].mean()) if len(daily) else 0.0,
        "max_weight_mean": float(daily["max_weight"].mean()) if len(daily) else 0.0,
        "gate_mean": float(daily["gate"].mean()) if len(daily) else 1.0,
    }
    summary["objective"] = (
        summary["ann_sharpe_like"]
        + 0.20 * summary["mean_ret"]
        + 0.10 * summary["positive_day_ratio"]
        + 0.50 * summary["max_drawdown_like"]
        - 0.03 * summary["turnover_mean"]
    )
    return ledger, daily, summary


def save_result(out_dir, name, ledger, daily, summary_rows, extra=None):
    out = ensure_dir(out_dir)
    ledger.to_csv(out / f"ledger_{name}.csv", index=False)
    daily.to_csv(out / f"daily_{name}.csv", index=False)
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out / f"summary_{name}.csv", index=False)
    if extra is not None:
        with open(out / f"report_{name}.json", "w", encoding="utf-8") as f:
            json.dump(extra, f, ensure_ascii=False, indent=2)
    print("===== SUMMARY =====", flush=True)
    print(summ.to_string(index=False), flush=True)
    print("[OK] saved:", out, flush=True)
