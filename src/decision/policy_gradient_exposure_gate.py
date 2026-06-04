#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V2X-RL+ Formal Exposure Controller

A more formal risk-aware RL / contextual-bandit exposure overlay.

Purpose:
- Do NOT learn stock selection.
- Only learn market exposure gate on top of V2X-B Cross Action Admission.
- Run multiple seeds and multiple action spaces.
- Report robustness instead of a single lucky run.

Protocol:
- Train/select on validation.
- Freeze policy and evaluate on test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_combined_or_daily(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "Date" not in df.columns:
        raise ValueError(f"No Date column in {p}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def daily_from_ledger(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])

    if "day_net_ret" in d.columns:
        out = d.groupby("Date", as_index=False).agg(
            base_ret=("day_net_ret", "first"),
            n_holding=("Ticker", "nunique") if "Ticker" in d.columns else ("Date", "size"),
            day_turnover=("day_turnover", "first") if "day_turnover" in d.columns else ("Date", "size"),
        )
        return out.sort_values("Date").reset_index(drop=True)

    if "ret" in d.columns and "gate" in d.columns and "base_ret" in d.columns:
        return d[["Date", "base_ret"]].drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

    if {"weight", "future_return_5d"}.issubset(d.columns):
        x = d["future_return_5d"].astype(float)
        if x.abs().quantile(0.999) > 1.0:
            d["future_return_5d"] = x / 100.0

        def f(g):
            return pd.Series({
                "base_ret": float((g["weight"].astype(float) * g["future_return_5d"].astype(float)).sum()),
                "n_holding": g["Ticker"].nunique() if "Ticker" in g.columns else len(g),
                "day_turnover": float(g["day_turnover"].iloc[0]) if "day_turnover" in g.columns else 1.0,
            })

        out = d.groupby("Date", group_keys=False).apply(f).reset_index()
        return out.sort_values("Date").reset_index(drop=True)

    raise ValueError("Cannot build daily returns from input.")


def split_ledger(path: str):
    raw = load_combined_or_daily(path)
    if "split" in raw.columns:
        val = raw[raw["split"].astype(str).str.lower().eq("val")].copy()
        test = raw[raw["split"].astype(str).str.lower().eq("test")].copy()
        if len(val) and len(test):
            return daily_from_ledger(val), daily_from_ledger(test)

    daily = daily_from_ledger(raw)
    # fallback by date
    val = daily[(daily["Date"] >= "2022-01-01") & (daily["Date"] <= "2023-12-31")].copy()
    test = daily[(daily["Date"] >= "2024-01-01")].copy()
    if len(val) and len(test):
        return val, test

    raise RuntimeError("Cannot split ledger into val/test. Need split column or Date ranges.")


def add_state(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy().sort_values("Date").reset_index(drop=True)
    r = d["base_ret"].astype(float)

    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0

    d["ret_lag1"] = r.shift(1).fillna(0)
    d["ret_lag2"] = r.shift(2).fillna(0)
    d["ret_lag3"] = r.shift(3).fillna(0)
    d["ma_3"] = r.shift(1).rolling(3, min_periods=1).mean().fillna(0)
    d["ma_5"] = r.shift(1).rolling(5, min_periods=1).mean().fillna(0)
    d["ma_10"] = r.shift(1).rolling(10, min_periods=1).mean().fillna(0)
    d["vol_5"] = r.shift(1).rolling(5, min_periods=2).std().fillna(0)
    d["vol_10"] = r.shift(1).rolling(10, min_periods=2).std().fillna(0)
    d["dd"] = dd.shift(1).fillna(0)
    d["dd_5_min"] = dd.shift(1).rolling(5, min_periods=1).min().fillna(0)

    neg = (r.shift(1).fillna(0) < 0).astype(int)
    d["neg_streak"] = neg.groupby((neg == 0).cumsum()).cumsum()

    raw_cols = [
        "ret_lag1", "ret_lag2", "ret_lag3",
        "ma_3", "ma_5", "ma_10",
        "vol_5", "vol_10",
        "dd", "dd_5_min", "neg_streak"
    ]

    for c in raw_cols:
        mu = d[c].expanding().mean().shift(1)
        sd = d[c].expanding().std().shift(1).replace(0, np.nan)
        d[c + "_z"] = ((d[c] - mu) / sd).replace([np.inf, -np.inf], 0).fillna(0).clip(-5, 5)

    return d


class Policy:
    def __init__(self, n_feat, n_act, seed):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.02, size=(n_feat, n_act))
        self.b = np.zeros(n_act)

    def probs(self, x):
        z = x @ self.W + self.b
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def sample(self, x, rng):
        p = self.probs(x)
        return int(rng.choice(len(p), p=p)), p

    def greedy(self, x):
        return int(np.argmax(self.probs(x)))

    def update(self, x, a, adv, lr, entropy_coef=0.0):
        p = self.probs(x)
        grad = -p
        grad[a] += 1.0
        self.W += lr * adv * np.outer(x, grad)
        self.b += lr * adv * grad

        if entropy_coef > 0:
            # small entropy-like smoothing toward uniform
            uni = np.ones_like(p) / len(p)
            self.b += entropy_coef * (uni - p)


def train_policy(val, actions, cfg, seed):
    feat_cols = [c for c in val.columns if c.endswith("_z")]
    X = val[feat_cols].to_numpy(float)
    R = val["base_ret"].to_numpy(float)

    policy = Policy(X.shape[1], len(actions), seed)
    rng = np.random.default_rng(seed)

    best = None
    best_obj = -1e18

    for ep in range(cfg["epochs"]):
        rewards = []
        traj = []

        equity = 1.0
        peak = 1.0
        prev_g = 1.0

        for t in range(len(val)):
            a, _ = policy.sample(X[t], rng)
            g = float(actions[a])
            ret = g * R[t]

            equity *= (1.0 + ret)
            peak = max(peak, equity)
            dd_abs = max(0.0, 1.0 - equity / peak)

            reward = ret
            reward -= cfg["dd_penalty"] * max(0.0, dd_abs - cfg["dd_trigger"])
            reward -= cfg["turnover_penalty"] * abs(g - prev_g)
            reward -= cfg["loss_penalty"] * max(0.0, -ret)

            traj.append((X[t], a, reward))
            rewards.append(reward)
            prev_g = g

        rewards = np.asarray(rewards)
        baseline = rewards.mean()
        scale = rewards.std() + 1e-8

        for x, a, reward in traj:
            adv = (reward - baseline) / scale
            policy.update(x, a, adv, cfg["lr"], cfg["entropy_coef"])

        if ep % 10 == 0 or ep == cfg["epochs"] - 1:
            _, s = evaluate(val, policy, actions, "val")
            obj = s["objective"]
            if obj > best_obj:
                best_obj = obj
                best = (policy.W.copy(), policy.b.copy())

    policy.W, policy.b = best
    return policy


def evaluate(daily, policy, actions, split):
    feat_cols = [c for c in daily.columns if c.endswith("_z")]
    X = daily[feat_cols].to_numpy(float)
    R = daily["base_ret"].to_numpy(float)

    gates = []
    rets = []

    for t in range(len(daily)):
        a = policy.greedy(X[t])
        g = float(actions[a])
        gates.append(g)
        rets.append(g * R[t])

    out = daily[["Date", "base_ret"]].copy()
    out["gate"] = gates
    out["ret"] = rets

    r = np.asarray(rets, dtype=float)
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0

    s = {
        "split": split,
        "n_days": int(len(r)),
        "mean_ret": float(np.mean(r)),
        "std_ret": float(np.std(r)),
        "sharpe_like": float(np.mean(r) / (np.std(r) + 1e-12)),
        "ann_ret_like": float(np.mean(r) * 252 / 5),
        "ann_sharpe_like": float(np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(252 / 5)),
        "max_drawdown_like": float(np.min(dd)),
        "positive_day_ratio": float(np.mean(r > 0)),
        "gate_mean": float(np.mean(gates)),
        "gate_min": float(np.min(gates)),
        "gate_max": float(np.max(gates)),
        "turnover_gate": float(np.mean(np.abs(np.diff(gates)))) if len(gates) > 1 else 0.0,
    }
    s["objective"] = (
        s["ann_sharpe_like"]
        + 0.5 * s["ann_ret_like"]
        + 1.0 * s["positive_day_ratio"]
        + 1.5 * s["max_drawdown_like"]
        - 0.2 * s["turnover_gate"]
    )
    return out, s


def fixed_exposure(daily, gate, split):
    out = daily[["Date", "base_ret"]].copy()
    out["gate"] = gate
    out["ret"] = out["base_ret"] * gate
    r = out["ret"].to_numpy(float)
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    s = {
        "split": split,
        "n_days": len(r),
        "mean_ret": float(np.mean(r)),
        "std_ret": float(np.std(r)),
        "sharpe_like": float(np.mean(r) / (np.std(r) + 1e-12)),
        "ann_ret_like": float(np.mean(r) * 252 / 5),
        "ann_sharpe_like": float(np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(252 / 5)),
        "max_drawdown_like": float(np.min(dd)),
        "positive_day_ratio": float(np.mean(r > 0)),
        "gate_mean": gate,
        "gate_min": gate,
        "gate_max": gate,
        "turnover_gate": 0.0,
    }
    s["objective"] = s["ann_sharpe_like"] + 0.5*s["ann_ret_like"] + s["positive_day_ratio"] + 1.5*s["max_drawdown_like"]
    return out, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="outputs/dive_trader_v2/v2x_overnight/61b_cross_action_admission/ledger_61b_cross_action_admission.csv")
    ap.add_argument("--out_dir", default="outputs/dive_trader_v2/v2x_rl_exposure_formal")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", default="2021,2022,2023,2024,2025,2026,2027,2028,2029,2030")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    val0, test0 = split_ledger(args.ledger)
    val = add_state(val0)
    test = add_state(test0)

    print("[DATA] val", val.shape, val["Date"].min(), val["Date"].max(), flush=True)
    print("[DATA] test", test.shape, test["Date"].min(), test["Date"].max(), flush=True)

    action_spaces = {
        "binary_025_100": np.array([0.25, 1.00]),
        "four_025_050_075_100": np.array([0.25, 0.50, 0.75, 1.00]),
        "five_020_040_060_080_100": np.array([0.20, 0.40, 0.60, 0.80, 1.00]),
        "conservative_025_050_100": np.array([0.25, 0.50, 1.00]),
    }

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    configs = []
    idx = 0
    for lr in [0.005, 0.01, 0.02]:
        for dd_penalty in [0.5, 1.0, 2.0]:
            for dd_trigger in [0.05, 0.08, 0.10]:
                for turnover_penalty in [0.0, 0.01]:
                    for loss_penalty in [0.0, 0.25]:
                        configs.append({
                            "idx": idx,
                            "lr": lr,
                            "dd_penalty": dd_penalty,
                            "dd_trigger": dd_trigger,
                            "turnover_penalty": turnover_penalty,
                            "loss_penalty": loss_penalty,
                            "entropy_coef": 0.000,
                            "epochs": args.epochs,
                        })
                        idx += 1

    all_rows = []
    best_records = []

    # fixed baselines
    for gate in [0.25, 0.50, 0.75, 1.00]:
        _, sv = fixed_exposure(val, gate, "val")
        _, st = fixed_exposure(test, gate, "test")
        for s in [sv, st]:
            s.update({"method": f"fixed_gate_{gate}", "action_space": "fixed", "seed": -1, "cfg_idx": -1})
            all_rows.append(s)

    for aname, actions in action_spaces.items():
        for seed in seeds:
            best_val = None
            best_test = None
            best_policy = None
            best_cfg = None

            for cfg in configs:
                policy = train_policy(val, actions, cfg, seed)
                _, sv = evaluate(val, policy, actions, "val")

                if best_val is None or sv["objective"] > best_val["objective"]:
                    _, st = evaluate(test, policy, actions, "test")
                    best_val = sv
                    best_test = st
                    best_policy = policy
                    best_cfg = cfg

            for split_s in [best_val, best_test]:
                split_s.update({
                    "method": "rl_exposure_formal",
                    "action_space": aname,
                    "seed": seed,
                    "cfg_idx": best_cfg["idx"],
                    "lr": best_cfg["lr"],
                    "dd_penalty": best_cfg["dd_penalty"],
                    "dd_trigger": best_cfg["dd_trigger"],
                    "turnover_penalty": best_cfg["turnover_penalty"],
                    "loss_penalty": best_cfg["loss_penalty"],
                })
                all_rows.append(split_s)

            daily_val, _ = evaluate(val, best_policy, actions, "val")
            daily_test, _ = evaluate(test, best_policy, actions, "test")
            daily_val.to_csv(out / f"daily_val_{aname}_seed{seed}.csv", index=False)
            daily_test.to_csv(out / f"daily_test_{aname}_seed{seed}.csv", index=False)

            best_records.append({
                "action_space": aname,
                "seed": seed,
                "val_objective": best_val["objective"],
                "test_objective": best_test["objective"],
                "test_ann_sharpe_like": best_test["ann_sharpe_like"],
                "test_max_drawdown_like": best_test["max_drawdown_like"],
                "test_mean_ret": best_test["mean_ret"],
                "test_gate_mean": best_test["gate_mean"],
                "cfg": best_cfg,
            })

            print(
                f"[DONE] {aname} seed={seed} "
                f"val_obj={best_val['objective']:.4f} "
                f"test_obj={best_test['objective']:.4f} "
                f"test_sharpe={best_test['ann_sharpe_like']:.4f} "
                f"test_dd={best_test['max_drawdown_like']:.4f}",
                flush=True,
            )

    res = pd.DataFrame(all_rows)
    res.to_csv(out / "all_results_v2x_rl_exposure_formal.csv", index=False)

    test_rl = res[(res["method"] == "rl_exposure_formal") & (res["split"] == "test")].copy()
    agg = test_rl.groupby("action_space").agg(
        n=("seed", "count"),
        mean_ret_mean=("mean_ret", "mean"),
        mean_ret_std=("mean_ret", "std"),
        ann_sharpe_mean=("ann_sharpe_like", "mean"),
        ann_sharpe_std=("ann_sharpe_like", "std"),
        maxdd_mean=("max_drawdown_like", "mean"),
        maxdd_std=("max_drawdown_like", "std"),
        positive_mean=("positive_day_ratio", "mean"),
        gate_mean=("gate_mean", "mean"),
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
    ).reset_index().sort_values("objective_mean", ascending=False)

    agg.to_csv(out / "summary_by_action_space_v2x_rl_exposure_formal.csv", index=False)

    best = pd.DataFrame(best_records).sort_values("test_objective", ascending=False)
    best.to_csv(out / "best_records_v2x_rl_exposure_formal.csv", index=False)

    with open(out / "report_v2x_rl_exposure_formal.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": "V2X-RL+ Formal Exposure Controller",
            "idea": "multi-seed action-space ablation for risk-aware exposure overlay; stock selection is fixed by V2X-B",
            "action_spaces": {k: v.tolist() for k, v in action_spaces.items()},
            "seeds": seeds,
            "n_configs": len(configs),
            "best_record": best.iloc[0].to_dict(),
        }, f, ensure_ascii=False, indent=2)

    print("===== ACTION SPACE SUMMARY TEST =====")
    print(agg.to_string(index=False))
    print("===== BEST RECORDS =====")
    print(best.head(20).to_string(index=False))
    print("[OK] saved:", out)


if __name__ == "__main__":
    main()
