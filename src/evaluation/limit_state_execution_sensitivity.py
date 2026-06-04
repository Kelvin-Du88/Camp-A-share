from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "outputs/dive_trader_v2/icdm_camp_final"
TAB = OUT / "tables"
FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data_external/msan_samples_full_qfq_norm"
LEDGER = ROOT / "outputs/dive_trader_v2/v2x_overnight/61b_cross_action_admission/ledger_test_61b_cross_action_admission.csv"
FINAL_DAILY = ROOT / "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure/daily_test_riemannian_ppo.csv"

META = DATA / "meta_L60_H5_full.parquet"
META2 = DATA / "meta_L60_H5_full_with_period_label.parquet"
X_CROSS = DATA / "X_cross_today_L60_H5_full.dat"
FEATURE_INFO = DATA / "feature_info_L60_H5_full.json"

TRADING_DAYS = 252
HORIZON = 5

def maxdd(r):
    nav = np.cumprod(1 + np.asarray(r, dtype=float))
    peak = np.maximum.accumulate(nav)
    return float((nav / np.maximum(peak, 1e-12) - 1).min())

def summarize(d, protocol, desc, renorm, haircut):
    r = d.groupby("Date")["ret_contrib"].sum().sort_index().to_numpy()
    mean = float(r.mean())
    std = float(r.std() + 1e-12)
    neg = r[r < 0]
    downside = float(neg.std() + 1e-12) if len(neg) else 1e-12
    return {
        "protocol": protocol,
        "model": "Action Admission Base",
        "n_days": int(len(r)),
        "mean_ret": mean,
        "std_ret": std,
        "ann_ret_like": mean * TRADING_DAYS / HORIZON,
        "ann_sharpe_like": mean / std * math.sqrt(TRADING_DAYS / HORIZON),
        "sortino_like": mean / downside * math.sqrt(TRADING_DAYS / HORIZON),
        "calmar_like": (mean * TRADING_DAYS / HORIZON) / abs(maxdd(r)) if abs(maxdd(r)) > 1e-12 else np.nan,
        "max_drawdown_like": maxdd(r),
        "worst_5pct_day_mean": float(np.mean(np.sort(r)[:max(1, int(len(r)*0.05))])),
        "positive_day_ratio": float((r > 0).mean()),
        "description": desc,
        "renorm": renorm,
        "haircut": haircut,
        "limit_weight_share_mean": float(d.groupby("Date").apply(lambda g: g.loc[g["limit_up_est"] > 0.5, "weight"].sum()).mean()),
        "one_line_like_weight_share_mean": float(d.groupby("Date").apply(lambda g: g.loc[g["one_line_like"] > 0.5, "weight"].sum()).mean()),
        "n_limit_mean": float(d.groupby("Date").apply(lambda g: (g["limit_up_est"] > 0.5).sum()).mean()),
        "n_one_line_like_mean": float(d.groupby("Date").apply(lambda g: (g["one_line_like"] > 0.5).sum()).mean()),
        "note": "Limit-up is not treated as mechanically untradable; one-line-like proxy and haircut rows are execution sensitivity diagnostics.",
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
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str)
    return df

info = json.loads(FEATURE_INFO.read_text())
cross_cols = info.get("cross_feature_cols", [])
need = ["limit_up_est", "limit_down_est", "broken_limit_est", "amount_rank", "turnover_rank", "volume_rank", "volume_ratio_rank"]
idx = {c: cross_cols.index(c) for c in need if c in cross_cols}

if "limit_up_est" not in idx:
    raise ValueError(f"limit_up_est not found in cross_feature_cols={cross_cols}")

meta_path = META2 if META2.exists() else META
meta = standardize_dt(pd.read_parquet(meta_path))
n = len(meta)

X = np.memmap(X_CROSS, dtype="float32", mode="r", shape=(n, len(cross_cols)))
feat = meta[["Date", "Ticker"]].copy()
for c, j in idx.items():
    feat[c] = np.asarray(X[:, j], dtype="float64")

ledger = pd.read_csv(LEDGER)
ledger = standardize_dt(ledger)

if FINAL_DAILY.exists():
    fd = pd.read_csv(FINAL_DAILY)
    fd["Date"] = pd.to_datetime(fd["Date"])
    ledger = ledger[ledger["Date"].isin(set(fd["Date"]))].copy()

if "future_return_5d" not in ledger.columns:
    raise ValueError("ledger missing future_return_5d.")
if "weight" not in ledger.columns:
    raise ValueError("ledger missing weight.")

base = ledger.merge(feat, on=["Date", "Ticker"], how="left")

for c in need:
    if c not in base.columns:
        base[c] = 0.0
base[need] = base[need].fillna(0.0)

# One-line-like proxy:
# high limit-up proxy + weak amount rank/turnover rank.
# Since amount_rank is usually [0,1] cross-sectional rank, low rank means lower liquidity.
base["one_line_like"] = (
    (base["limit_up_est"] > 0.5)
    & (base["amount_rank"] < 0.20)
    & (base["turnover_rank"] < 0.35)
).astype(float)

base["ret_contrib_base"] = base["weight"].astype(float) * base["future_return_5d"].astype(float)

def apply_protocol(df, protocol, desc, mask_col=None, exclude=False, haircut=1.0, renorm=False):
    d = df.copy()
    d["adj_weight"] = d["weight"].astype(float)

    if mask_col is not None:
        mask = d[mask_col].astype(float) > 0.5
        if exclude:
            d.loc[mask, "adj_weight"] = 0.0
        else:
            d.loc[mask, "adj_weight"] = d.loc[mask, "adj_weight"] * haircut

    if renorm:
        old_sum = d.groupby("Date")["weight"].transform("sum")
        new_sum = d.groupby("Date")["adj_weight"].transform("sum")
        scale = old_sum / new_sum.replace(0, np.nan)
        d["adj_weight"] = (d["adj_weight"] * scale).replace([np.inf, -np.inf], 0).fillna(0)

    d["ret_contrib"] = d["adj_weight"] * d["future_return_5d"].astype(float)

    return summarize(d, protocol, desc, renorm, haircut)

rows = []
rows.append(apply_protocol(base, "include_all", "include all selected names", None, False, 1.0, False))
rows.append(apply_protocol(base, "one_line_like_haircut_50", "one-line-like proxy weight haircut 50%, no renorm", "one_line_like", False, 0.5, False))
rows.append(apply_protocol(base, "one_line_like_haircut_20", "one-line-like proxy weight haircut 80%, no renorm", "one_line_like", False, 0.2, False))
rows.append(apply_protocol(base, "exclude_one_line_like_no_renorm", "exclude one-line-like proxy, no renorm", "one_line_like", True, 1.0, False))
rows.append(apply_protocol(base, "exclude_one_line_like_renorm", "exclude one-line-like proxy, renorm", "one_line_like", True, 1.0, True))
rows.append(apply_protocol(base, "limit_up_haircut_50", "all limit-up proxy weight haircut 50%, no renorm", "limit_up_est", False, 0.5, False))
rows.append(apply_protocol(base, "limit_up_haircut_20", "all limit-up proxy weight haircut 80%, no renorm", "limit_up_est", False, 0.2, False))
rows.append(apply_protocol(base, "exclude_limit_up_no_renorm", "exclude all limit-up proxy, stress upper bound", "limit_up_est", True, 1.0, False))
rows.append(apply_protocol(base, "exclude_limit_up_renorm", "exclude all limit-up proxy, renorm", "limit_up_est", True, 1.0, True))

out = pd.DataFrame(rows)
out.to_csv(TAB / "table_117_limit_state_execution_sensitivity.csv", index=False)
out.round(6).to_csv(TAB / "table_117_limit_state_execution_sensitivity_rounded.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
plot = out[["protocol", "mean_ret", "max_drawdown_like"]].copy()
ax.bar(plot["protocol"], plot["mean_ret"])
ax.set_ylabel("Mean return-like")
ax.set_title("Limit-state execution sensitivity")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(FIG / "fig_117_limit_state_execution_sensitivity.pdf")
plt.savefig(FIG / "fig_117_limit_state_execution_sensitivity.png", dpi=240)
plt.close()

base[[
    "Date","Ticker","weight","future_return_5d","limit_up_est","limit_down_est",
    "broken_limit_est","amount_rank","turnover_rank","one_line_like"
]].to_csv(TAB / "daily_selected_limit_state_features_117.csv", index=False)

print(out.to_string(index=False))
print("[OK]", TAB / "table_117_limit_state_execution_sensitivity.csv")
