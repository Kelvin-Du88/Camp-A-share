from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "outputs/dive_trader_v2/icdm_camp_final"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data_external/msan_samples_full_qfq_norm"
META_CANDIDATES = [
    DATA / "meta_L60_H5_full_with_period_label.parquet",
    DATA / "meta_L60_H5_full.parquet",
]
YRET = DATA / "y_ret_5d_L60_H5_full.dat"

SCORES = {
    "Temporal": "outputs/dive_trader_v2/experts/v2l_frequency_expert/score_test_v2l_frequency_expert.csv",
    "FactorTree": "outputs/dive_trader_v2/experts/factor_tree/score_test_factor_tree.csv",
    "RankICLinear": "outputs/dive_trader_v2/experts/v2j_rankic_linear/score_test_v2j_rankic_linear.csv",
}

def standardize_dt(df):
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["date", "trade_date", "datetime"]:
            rename[c] = "Date"
        if lc in ["ticker", "ts_code", "symbol", "code", "stock_code"]:
            rename[c] = "Ticker"
    df = df.rename(columns=rename)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str)
    return df

def find_score_col(df):
    if "score" in df.columns:
        return "score"
    for c in df.columns:
        lc = c.lower()
        if "score" in lc or "pred" in lc or "alpha" in lc:
            return c
    return None

def load_target():
    meta_path = next((p for p in META_CANDIDATES if p.exists()), None)
    if meta_path is None:
        raise FileNotFoundError("No meta parquet found.")

    m = pd.read_parquet(meta_path)
    m = standardize_dt(m)

    if "Date" not in m.columns or "Ticker" not in m.columns:
        raise ValueError(f"meta missing Date/Ticker. columns={list(m.columns)[:80]}")

    target_col = None
    for c in [
        "future_return_5d",
        "future_ret_5d",
        "y_ret_5d",
        "ret_5d",
        "target_ret_5d",
        "label_ret_5d",
    ]:
        if c in m.columns:
            target_col = c
            break

    out = m[["Date", "Ticker"]].copy()

    if target_col is not None:
        out["target_ret_5d"] = m[target_col].astype(float).to_numpy()
        source = f"meta:{target_col}"
    else:
        n = len(m)
        if not YRET.exists():
            raise FileNotFoundError(f"Cannot find target memmap: {YRET}")
        y = np.memmap(YRET, dtype="float32", mode="r", shape=(n,))
        out["target_ret_5d"] = np.asarray(y, dtype="float64")
        source = "memmap:y_ret_5d_L60_H5_full.dat"

    return out, source, str(meta_path)

def daily_rankic(g):
    if g["score"].nunique() < 2 or g["target_ret_5d"].nunique() < 2:
        return np.nan
    return g["score"].rank().corr(g["target_ret_5d"].rank())

def p_at_k(g, k):
    s = g.sort_values("score", ascending=False).head(k)
    return float((s["target_ret_5d"] > 0).mean()), float(s["target_ret_5d"].mean())

target, target_source, meta_source = load_target()
rows = []

for model, rel in SCORES.items():
    p = ROOT / rel
    if not p.exists():
        rows.append({"model": model, "status": "missing_score_file", "source": rel})
        continue

    s = pd.read_csv(p)
    s = standardize_dt(s)
    score_col = find_score_col(s)

    if not {"Date", "Ticker"}.issubset(s.columns) or score_col is None:
        rows.append({
            "model": model,
            "status": "missing_required_cols",
            "source": rel,
            "columns": "|".join(s.columns),
        })
        continue

    s = s[["Date", "Ticker", score_col]].rename(columns={score_col: "score"})
    s["Ticker"] = s["Ticker"].astype(str)

    df = s.merge(target, on=["Date", "Ticker"], how="inner")

    if df.empty:
        # fallback: sometimes ticker has suffix mismatch. Try normalized numeric part.
        s2 = s.copy()
        t2 = target.copy()
        s2["Ticker_key"] = s2["Ticker"].str.replace(r"\D", "", regex=True).str[-6:]
        t2["Ticker_key"] = t2["Ticker"].str.replace(r"\D", "", regex=True).str[-6:]
        df = s2.merge(t2[["Date", "Ticker_key", "target_ret_5d"]], on=["Date", "Ticker_key"], how="inner")

    if df.empty:
        rows.append({
            "model": model,
            "status": "merge_empty",
            "source": rel,
            "score_rows": len(s),
            "target_rows": len(target),
            "target_source": target_source,
            "meta_source": meta_source,
        })
        continue

    ics, p10, r10, p20, r20, p30, r30 = [], [], [], [], [], [], []

    for _, g in df.groupby("Date", sort=True):
        if len(g) < 30:
            continue
        ic = daily_rankic(g)
        if not np.isnan(ic):
            ics.append(ic)

        a, b = p_at_k(g, 10); p10.append(a); r10.append(b)
        a, b = p_at_k(g, 20); p20.append(a); r20.append(b)
        a, b = p_at_k(g, 30); p30.append(a); r30.append(b)

    if len(ics) == 0:
        rows.append({
            "model": model,
            "status": "no_valid_days",
            "source": rel,
            "n_pairs": len(df),
            "target_source": target_source,
            "meta_source": meta_source,
        })
        continue

    ics = np.asarray(ics, dtype=float)

    rows.append({
        "model": model,
        "status": "ready",
        "source": rel,
        "target_source": target_source,
        "meta_source": meta_source,
        "n_pairs": int(len(df)),
        "n_days": int(len(ics)),
        "rankic_mean": float(np.nanmean(ics)),
        "rankic_std": float(np.nanstd(ics)),
        "rankic_tstat": float(np.nanmean(ics) / (np.nanstd(ics) + 1e-12) * np.sqrt(max(1, len(ics)))),
        "precision_at_10": float(np.mean(p10)),
        "top10_mean_ret": float(np.mean(r10)),
        "precision_at_20": float(np.mean(p20)),
        "top20_mean_ret": float(np.mean(r20)),
        "precision_at_30": float(np.mean(p30)),
        "top30_mean_ret": float(np.mean(r30)),
        "protocol": "score files merged with target by Date,Ticker; fallback normalized ticker key; daily RankIC and Precision@K",
    })

out = pd.DataFrame(rows)
out.to_csv(TAB / "table_115_rankic_precision_k_merged.csv", index=False)
out.round(6).to_csv(TAB / "table_115_rankic_precision_k_merged_rounded.csv", index=False)

print(out.to_string(index=False))
print("[OK]", TAB / "table_115_rankic_precision_k_merged.csv")
