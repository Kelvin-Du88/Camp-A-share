from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("outputs/dive_trader_v2/kdd_defense_pack_v2")
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

def read_csv(p):
    p = Path(p)
    if not p.exists():
        print("[MISS]", p)
        return None
    return pd.read_csv(p)

def norm_date(df):
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

def maxdd(ret):
    ret = pd.Series(ret, dtype="float64").fillna(0)
    nav = (1 + ret).cumprod()
    dd = nav / nav.cummax() - 1
    return float(dd.min())

def metrics(ret):
    ret = pd.Series(ret, dtype="float64").dropna()
    std = ret.std(ddof=0)
    mean = ret.mean()
    return {
        "n_days": len(ret),
        "mean_ret": mean,
        "std_ret": std,
        "ann_ret_like": mean * 252 / 5,
        "ann_sharpe_like": mean / (std + 1e-12) * np.sqrt(252 / 5),
        "max_drawdown_like": maxdd(ret),
        "positive_day_ratio": (ret > 0).mean(),
    }

# Load base and RL
base_ledger = norm_date(pd.read_csv("outputs/dive_trader_v2/v2x_overnight/61b_cross_action_admission/ledger_test_61b_cross_action_admission.csv"))
base = base_ledger.groupby("Date", as_index=False).agg(
    base_ret=("day_net_ret", "first"),
    turnover=("day_turnover", "first"),
)
rl = norm_date(pd.read_csv("outputs/dive_trader_v2/v2x_rl_exposure_formal/daily_test_five_020_040_060_080_100_seed2022.csv"))
if "ret" not in rl.columns:
    raise SystemExit("RL daily file lacks ret")
if "gate" not in rl.columns:
    rl["gate"] = 1.0
rl = rl[["Date", "ret", "gate"]].rename(columns={"ret": "rl_ret"})

df = base.merge(rl, on="Date", how="inner")

# Load market emotion / index features
candidates = [
    "data_external/market_emotion_features/daily_market_emotion_plus_clean.parquet",
    "data_external/market_emotion_features/daily_market_emotion_plus.parquet",
    "data_external/market_emotion_features/index_daily_features.parquet",
]
mkt = None
for p in candidates:
    pp = Path(p)
    if pp.exists():
        if pp.suffix == ".parquet":
            mkt = pd.read_parquet(pp)
        else:
            mkt = pd.read_csv(pp)
        print("[LOAD MKT]", pp, mkt.shape)
        break

if mkt is not None:
    mkt = norm_date(mkt)
    useful = [
        "Date",
        "ret_mean", "ret_std", "ret_median", "ret_p05", "ret_p10",
        "up_ratio", "down_ratio",
        "limit_up_count", "limit_down_count", "broken_limit_count", "broken_limit_rate",
        "amount_sum_chg1_pct", "market_heat_score", "market_risk_score",
        "emotion_raw", "emotion_ma5", "emotion_accel",
        "idx_csi300_ret", "idx_csi500_ret", "idx_csi1000_ret",
        "idx_chinext_ret", "idx_shanghai_composite_ret", "idx_shenzhen_component_ret",
    ]
    cols = [c for c in useful if c in mkt.columns]
    df = df.merge(mkt[cols], on="Date", how="left")

# Failure cases
df["protection"] = df["rl_ret"] - df["base_ret"]
df["protected"] = df["protection"] > 0
df["base_drawdown"] = (1 + df["base_ret"]).cumprod() / (1 + df["base_ret"]).cumprod().cummax() - 1
df["rl_drawdown"] = (1 + df["rl_ret"]).cumprod() / (1 + df["rl_ret"]).cumprod().cummax() - 1
df["base_ret_ma5"] = df["base_ret"].rolling(5).mean()
df["base_vol_ma5"] = df["base_ret"].rolling(5).std()
df["base_neg_streak"] = (df["base_ret"] < 0).astype(int).groupby((df["base_ret"] >= 0).astype(int).cumsum()).cumsum()

def reason(row):
    reasons = []
    if row.get("gate", 1.0) >= 1.0:
        reasons.append("full exposure")
    else:
        reasons.append("reduced exposure")
    if pd.notna(row.get("idx_csi1000_ret", np.nan)) and row.get("idx_csi1000_ret") < -0.02:
        reasons.append("small-cap index shock")
    if pd.notna(row.get("ret_p10", np.nan)) and row.get("ret_p10") < -0.03:
        reasons.append("broad market left-tail")
    if pd.notna(row.get("down_ratio", np.nan)) and row.get("down_ratio") > 0.60:
        reasons.append("broad down day")
    if pd.notna(row.get("limit_down_count", np.nan)) and row.get("limit_down_count") > 50:
        reasons.append("limit-down pressure")
    if pd.notna(row.get("market_risk_score", np.nan)) and row.get("market_risk_score") > 0.7:
        reasons.append("high market risk score")
    return "; ".join(reasons)

df["failure_reason_proxy"] = df.apply(reason, axis=1)

worst_rl = df.nsmallest(20, "rl_ret").copy()
worst_base = df.nsmallest(20, "base_ret").copy()
worst_rl.to_csv(TAB / "table_9a_worst_rl_failure_attribution.csv", index=False)
worst_base.to_csv(TAB / "table_9b_worst_base_failure_attribution.csv", index=False)

# Summary by gate
gate_summary = df.groupby("gate", dropna=False).agg(
    n_days=("Date", "count"),
    mean_base_ret=("base_ret", "mean"),
    mean_rl_ret=("rl_ret", "mean"),
    mean_protection=("protection", "mean"),
    min_rl_ret=("rl_ret", "min"),
    mean_base_drawdown=("base_drawdown", "mean"),
    mean_rl_drawdown=("rl_drawdown", "mean"),
).reset_index()
gate_summary["ratio"] = gate_summary["n_days"] / len(df)
gate_summary.to_csv(TAB / "table_9c_gate_state_return_summary.csv", index=False)

# Tail-aware diagnostic override, not final method
# simple rule: if recent base drawdown/vol/market left-tail worsens, cap exposure.
dd5 = (1 + df["base_ret"]).rolling(5).apply(lambda x: np.prod(x)-1, raw=False)
df["tail_flag"] = False
df.loc[df["base_vol_ma5"] > df["base_vol_ma5"].quantile(0.80), "tail_flag"] = True
if "ret_p10" in df.columns:
    df.loc[df["ret_p10"] < df["ret_p10"].quantile(0.20), "tail_flag"] = True
if "market_risk_score" in df.columns:
    df.loc[df["market_risk_score"] > df["market_risk_score"].quantile(0.80), "tail_flag"] = True

df["tail_gate"] = df["gate"].where(~df["tail_flag"], np.minimum(df["gate"], 0.8))
df["tail_ret"] = df["base_ret"] * df["tail_gate"]

summary = pd.DataFrame([
    {"model": "V2X-B", **metrics(df["base_ret"])},
    {"model": "V2X-RL", **metrics(df["rl_ret"])},
    {"model": "V2X-RL-tail-diagnostic", **metrics(df["tail_ret"])},
])
summary["gate_mean"] = [1.0, df["gate"].mean(), df["tail_gate"].mean()]
summary.to_csv(TAB / "table_10_tail_diagnostic_overlay.csv", index=False)

# Figures
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["base_drawdown"], label="V2X-B drawdown", linewidth=1.5)
plt.plot(df["Date"], df["rl_drawdown"], label="V2X-RL drawdown", linewidth=1.5)
plt.scatter(worst_rl["Date"], worst_rl["rl_drawdown"], s=22, label="Worst RL days")
plt.title("Failure Attribution: Worst RL Days on Drawdown Curve")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True, alpha=0.25)
for ext in ["png", "pdf"]:
    plt.savefig(FIG / f"fig_9a_failure_days_drawdown.{ext}", bbox_inches="tight", dpi=240)
plt.close()

if "idx_csi1000_ret" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["idx_csi1000_ret"], df["rl_ret"], s=14, alpha=0.6)
    plt.scatter(worst_rl["idx_csi1000_ret"], worst_rl["rl_ret"], s=38, label="Worst RL days")
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("CSI1000 daily return")
    plt.ylabel("V2X-RL return")
    plt.title("Failure Days vs Small-cap Market State")
    plt.legend()
    plt.grid(True, alpha=0.25)
    for ext in ["png", "pdf"]:
        plt.savefig(FIG / f"fig_9b_failure_vs_csi1000.{ext}", bbox_inches="tight", dpi=240)
    plt.close()

plt.figure(figsize=(8, 5))
plt.bar(summary["model"], summary["max_drawdown_like"])
plt.ylabel("Max drawdown-like")
plt.title("Tail Diagnostic Overlay: Drawdown Comparison")
plt.xticks(rotation=18, ha="right")
plt.grid(True, axis="y", alpha=0.25)
for ext in ["png", "pdf"]:
    plt.savefig(FIG / f"fig_10_tail_diagnostic_maxdd.{ext}", bbox_inches="tight", dpi=240)
plt.close()

status = []
for p in sorted(list(FIG.glob("*")) + list(TAB.glob("*"))):
    status.append({"file": str(p), "size_kb": round(p.stat().st_size/1024, 2)})
status = pd.DataFrame(status)
status.to_csv(OUT / "status_v2.csv", index=False)

print("===== failure worst rl =====")
print(worst_rl[["Date","base_ret","rl_ret","gate","protection","failure_reason_proxy"]].to_string(index=False))

print("\n===== gate summary =====")
print(gate_summary.to_string(index=False))

print("\n===== tail diagnostic =====")
print(summary.to_string(index=False))

print("\n===== status =====")
print(status.to_string(index=False))
