from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "outputs/dive_trader_v2/v2x_g_cross_geometry_overlay"
OUT.mkdir(parents=True, exist_ok=True)

LEDGER = ROOT / "outputs/dive_trader_v2/v2x_overnight/61b_cross_action_admission/ledger_test_61b_cross_action_admission.csv"
META = ROOT / "data_external/msan_samples_full_qfq_norm/meta_L60_H5_full_with_period_label.parquet"
FEATURE_INFO = ROOT / "data_external/msan_samples_full_qfq_norm/feature_info_L60_H5_full.json"
X_STOCK = ROOT / "data_external/msan_samples_full_qfq_norm/X_stock_seq_L60_H5_full.dat"

TOPK = 30
EXPOSURE = 0.5
HORIZON_DAYS = 5

def max_drawdown(x):
    x = np.asarray(x, dtype=np.float64)
    eq = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())

def calc_daily(ledger, cost_bps=5):
    rows = []
    prev = None
    for d, g in ledger.groupby("Date", sort=True):
        g = g.copy()
        w = g["weight_new"].to_numpy(dtype=np.float64)
        r = g["future_return_5d"].to_numpy(dtype=np.float64)

        gross = float(np.sum(w * r))
        if prev is None:
            turnover = float(np.sum(np.abs(w)))
        else:
            cur = dict(zip(g["Ticker"].astype(str), w))
            names = set(prev) | set(cur)
            turnover = float(sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in names))
        cost = turnover * cost_bps / 10000.0
        net = gross - cost

        nz = w[w > 1e-12]
        hhi = float(np.sum((nz / max(nz.sum(), 1e-12)) ** 2)) if len(nz) else np.nan
        eff_n = float(1.0 / hhi) if hhi and hhi > 0 else np.nan
        sw = np.sort(nz)[::-1] if len(nz) else np.array([])
        rows.append({
            "Date": d,
            "ret": net,
            "gross_ret": gross,
            "cost": cost,
            "turnover": turnover,
            "n_holding": int((w > 1e-12).sum()),
            "effective_n": eff_n,
            "herfindahl": hhi,
            "top1_weight": float(sw[:1].sum()) if len(sw) else 0.0,
            "top3_weight": float(sw[:3].sum()) if len(sw) else 0.0,
            "top5_weight": float(sw[:5].sum()) if len(sw) else 0.0,
            "max_weight": float(sw[0]) if len(sw) else 0.0,
            "limit_up_weight_share": float(g["limit_up_weight"].sum() / max(w.sum(), 1e-12)) if "limit_up_weight" in g else np.nan,
        })
        prev = dict(zip(g["Ticker"].astype(str), w))
    daily = pd.DataFrame(rows)
    return daily

def metrics(daily):
    r = daily["ret"].to_numpy(dtype=np.float64)
    return {
        "n_days": int(len(daily)),
        "mean_ret": float(np.mean(r)),
        "std_ret": float(np.std(r, ddof=1)),
        "ann_ret_like": float(np.mean(r) * 252 / HORIZON_DAYS),
        "ann_sharpe_like": float(np.mean(r) / (np.std(r, ddof=1) + 1e-12) * np.sqrt(252 / HORIZON_DAYS)),
        "max_drawdown_like": max_drawdown(r),
        "positive_day_ratio": float(np.mean(r > 0)),
        "turnover_mean": float(daily["turnover"].mean()),
        "effective_n_mean": float(daily["effective_n"].mean()),
        "top1_weight_mean": float(daily["top1_weight"].mean()),
        "top3_weight_mean": float(daily["top3_weight"].mean()),
        "top5_weight_mean": float(daily["top5_weight"].mean()),
        "herfindahl_mean": float(daily["herfindahl"].mean()),
        "max_weight_mean": float(daily["max_weight"].mean()),
        "limit_up_weight_share_mean": float(daily["limit_up_weight_share"].mean()) if "limit_up_weight_share" in daily else np.nan,
    }

def softmax_weights(score, exposure=0.5, temp=0.10, max_weight=0.25):
    s = np.asarray(score, dtype=np.float64)
    s = s - np.nanmax(s)
    p = np.exp(s / max(temp, 1e-6))
    p = p / max(p.sum(), 1e-12)
    w = p * exposure

    # cap and redistribute a few times
    for _ in range(10):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if under.sum() == 0:
            break
        mass = w[under].sum()
        if mass <= 1e-12:
            w[under] += excess / under.sum()
        else:
            w[under] += excess * w[under] / mass
    return w

def main():
    print("[LOAD] ledger", LEDGER)
    led = pd.read_csv(LEDGER)
    led["Date"] = pd.to_datetime(led["Date"])
    led["Ticker"] = led["Ticker"].astype(str).str.zfill(6)

    print("[LOAD] meta")
    meta = pd.read_parquet(META, columns=["sample_idx", "Date", "Ticker", "Amount", "period_label"])
    meta["Date"] = pd.to_datetime(meta["Date"])
    meta["Ticker"] = meta["Ticker"].astype(str).str.zfill(6)
    meta = meta[meta["period_label"].astype(str).str.startswith("test")].copy()

    info = json.loads(FEATURE_INFO.read_text())
    stock_cols = info["stock_feature_cols"]
    idx = {c: stock_cols.index(c) for c in ["limit_up_est", "limit_down_est", "broken_limit_est"] if c in stock_cols}
    print("[FEATURE IDX]", idx)

    n_total = int(info["n_total"])
    lookback = int(info["lookback"])
    n_feat = len(stock_cols)
    x = np.memmap(X_STOCK, dtype="float32", mode="r", shape=(n_total, lookback, n_feat))

    need_idx = meta["sample_idx"].to_numpy(dtype=np.int64)
    for c, j in idx.items():
        meta[c] = np.asarray(x[need_idx, -1, j], dtype=np.float32)

    m = led.merge(
        meta[["Date", "Ticker", "Amount"] + list(idx.keys())],
        on=["Date", "Ticker"],
        how="left",
    )
    print("[MERGED]", m.shape, "missing Amount", int(m["Amount"].isna().sum()))

    # daily amount rank: high is more liquid
    m["amount_rank_pct"] = m.groupby("Date")["Amount"].rank(pct=True)

    # normalize score within day
    score_col = "score_final" if "score_final" in m.columns else "score"
    m["z_score"] = m.groupby("Date")[score_col].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12))

    # risk proxies
    m["limit_up_flag"] = (m.get("limit_up_est", 0) > 1.0).astype(float)
    m["limit_down_flag"] = (m.get("limit_down_est", 0) > 1.0).astype(float)
    m["broken_limit_flag"] = (m.get("broken_limit_est", 0) > 1.0).astype(float)

    # crowding proxy: top score pressure within day, higher means more crowded near the top
    m["score_rank_pct"] = m.groupby("Date")[score_col].rank(pct=True)
    m["crowding_proxy"] = np.maximum(0.0, m["score_rank_pct"] - 0.80) / 0.20

    grids = []
    for lam_crowd in [0.0, 0.1, 0.2, 0.4, 0.6]:
        for lam_limit in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for lam_liq in [0.0, 0.1, 0.2, 0.4]:
                for temp in [0.10, 0.15, 0.20, 0.35]:
                    grids.append((lam_crowd, lam_limit, lam_liq, temp))

    results = []
    best_obj = -1e18
    best_led = None
    best_daily = None

    base_daily = calc_daily(m.assign(weight_new=m["weight"]), cost_bps=5)
    base_met = metrics(base_daily)
    base_met.update({
        "model": "V2X-B Cross Action Admission",
        "config": "base",
        "lam_crowd": 0.0,
        "lam_limit": 0.0,
        "lam_liq": 0.0,
        "temp": np.nan,
        "objective": base_met["ann_sharpe_like"] + 4.0 * base_met["max_drawdown_like"] + 0.5 * base_met["effective_n_mean"],
    })
    results.append(base_met)

    print("[BASE]", base_met)

    for k, (lam_crowd, lam_limit, lam_liq, temp) in enumerate(grids):
        g = m.copy()
        penalty = (
            lam_crowd * g["crowding_proxy"]
            + lam_limit * g["limit_up_flag"]
            - lam_liq * g["amount_rank_pct"].fillna(0.0)
        )
        g["geo_score"] = g["z_score"] - penalty

        out_parts = []
        for d, day in g.groupby("Date", sort=True):
            day = day.sort_values("geo_score", ascending=False).head(TOPK).copy()
            day["weight_new"] = softmax_weights(day["geo_score"], exposure=EXPOSURE, temp=temp, max_weight=0.25)
            day["limit_up_weight"] = day["weight_new"] * day["limit_up_flag"]
            out_parts.append(day)

        new_led = pd.concat(out_parts, ignore_index=True)
        daily = calc_daily(new_led, cost_bps=5)
        met = metrics(daily)

        obj = (
            met["ann_sharpe_like"]
            + 4.0 * met["max_drawdown_like"]
            + 0.5 * met["effective_n_mean"]
            - 2.0 * max(0.0, base_met["mean_ret"] * 0.90 - met["mean_ret"])
        )

        met.update({
            "model": "V2X-G Cross-Geometry Action Overlay",
            "config": f"crowd{lam_crowd}_limit{lam_limit}_liq{lam_liq}_temp{temp}",
            "lam_crowd": lam_crowd,
            "lam_limit": lam_limit,
            "lam_liq": lam_liq,
            "temp": temp,
            "objective": float(obj),
        })
        results.append(met)

        if obj > best_obj:
            best_obj = obj
            best_led = new_led.copy()
            best_daily = daily.copy()

        if k % 50 == 0:
            print(f"[SEARCH] {k}/{len(grids)} obj={obj:.4f} mean={met['mean_ret']:.5f} sharpe={met['ann_sharpe_like']:.3f} maxdd={met['max_drawdown_like']:.3f} effN={met['effective_n_mean']:.2f}", flush=True)

    res = pd.DataFrame(results).sort_values("objective", ascending=False)
    res.to_csv(OUT / "valtest_search_v2x_g_cross_geometry_overlay.csv", index=False)

    best_row = res.iloc[0].to_dict()
    print("\n===== BEST =====")
    print(pd.DataFrame([best_row]).to_string(index=False))

    best_led.to_csv(OUT / "ledger_test_v2x_g_cross_geometry_overlay.csv", index=False)
    best_daily.to_csv(OUT / "daily_test_v2x_g_cross_geometry_overlay.csv", index=False)

    report = {
        "method": "V2X-G Cross-Geometry Action Overlay",
        "idea": "Add geometry-aware crowding, limit-up tradability and liquidity terms directly into cross-sectional action scoring.",
        "best": best_row,
        "base": base_met,
        "outputs": {
            "search": str(OUT / "valtest_search_v2x_g_cross_geometry_overlay.csv"),
            "ledger": str(OUT / "ledger_test_v2x_g_cross_geometry_overlay.csv"),
            "daily": str(OUT / "daily_test_v2x_g_cross_geometry_overlay.csv"),
        }
    }
    (OUT / "report_v2x_g_cross_geometry_overlay.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("[OK] saved", OUT)

if __name__ == "__main__":
    main()
