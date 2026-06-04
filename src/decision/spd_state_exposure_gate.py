from pathlib import Path
import json
import math
import random
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(".")
OUT = ROOT / "outputs/dive_trader_v2/v2x_riemannian_ppo_exposure"
OUT.mkdir(parents=True, exist_ok=True)

DATA_DIR = ROOT / "data_external/msan_samples_full_qfq_norm"

BASE_LEDGER = ROOT / "outputs/dive_trader_v2/v2x_overnight/61b_cross_action_admission/ledger_test_61b_cross_action_admission.csv"
BASE_LEDGER_VAL = ROOT / "outputs/dive_trader_v2/v2x_overnight/61b_cross_action_admission/ledger_val_61b_cross_action_admission.csv"

OLD_RL_TEST = ROOT / "outputs/dive_trader_v2/v2x_rl_exposure_formal/daily_test_five_020_040_060_080_100_seed2022.csv"

HORIZON_DAYS = 5
TRADING_DAYS = 252
COST_BPS = 5.0

ACTIONS = np.array([0.20, 0.40, 0.60, 0.80, 1.00], dtype=np.float64)

RNG_SEEDS = [2021, 2022, 2023, 2024, 2025]


def max_drawdown(ret):
    ret = np.asarray(ret, dtype=np.float64)
    nav = np.cumprod(1.0 + ret)
    peak = np.maximum.accumulate(nav)
    dd = nav / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def summarize_daily(daily, ret_col="ret", name="model"):
    r = daily[ret_col].astype(float).to_numpy()
    mean = float(np.mean(r))
    std = float(np.std(r) + 1e-12)
    sharpe = mean / std
    return {
        "model": name,
        "n_days": int(len(daily)),
        "mean_ret": mean,
        "std_ret": std,
        "ann_ret_like": mean * TRADING_DAYS / HORIZON_DAYS,
        "ann_sharpe_like": sharpe * math.sqrt(TRADING_DAYS / HORIZON_DAYS),
        "max_drawdown_like": max_drawdown(r),
        "positive_day_ratio": float((r > 0).mean()),
    }


def read_daily_from_ledger(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    if "day_net_ret" in df.columns:
        d = (
            df.groupby("Date", as_index=False)
            .agg(
                base_ret=("day_net_ret", "first"),
                day_turnover=("day_turnover", "first"),
                day_all_ret=("day_all_ret", "first"),
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )
    else:
        d = (
            df.groupby("Date", as_index=False)
            .apply(lambda g: pd.Series({
                "base_ret": float((g["weight"].astype(float) * g["future_return_5d"].astype(float)).sum()),
                "day_turnover": np.nan,
                "day_all_ret": float(g["future_return_5d"].astype(float).mean()),
            }))
            .reset_index(drop=True)
            .sort_values("Date")
        )
    return d


def safe_spd(mat, eps=1e-4):
    mat = np.asarray(mat, dtype=np.float64)
    mat = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T


def logm_spd(mat, eps=1e-8):
    mat = safe_spd(mat, eps=eps)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return (vecs * np.log(vals)) @ vecs.T


def invsqrt_spd(mat, eps=1e-8):
    mat = safe_spd(mat, eps=eps)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def affine_invariant_distance(a, b):
    a = safe_spd(a)
    b = safe_spd(b)
    a_inv_sqrt = invsqrt_spd(a)
    c = a_inv_sqrt @ b @ a_inv_sqrt
    lc = logm_spd(c)
    return float(np.linalg.norm(lc, ord="fro"))


def load_market_state():
    p = DATA_DIR / "market_seq_date_table_L60_H5_full.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    tab = pd.read_parquet(p)
    tab["Date"] = pd.to_datetime(tab["Date"])
    return tab


def infer_market_seq_path():
    candidates = [
        DATA_DIR / "X_market_seq_by_date_L60_H5_full.npy",
        DATA_DIR / "X_market_seq_by_date_L60_H5_full.dat",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find X_market_seq_by_date file.")


def compute_riemannian_daily_state():
    print("[STEP] compute SPD-Riemannian daily market state", flush=True)

    date_tab = load_market_state()
    path = infer_market_seq_path()

    if path.suffix == ".npy":
        X = np.load(path, mmap_mode="r")
    else:
        # fallback shape from project memory
        X = np.memmap(path, dtype="float32", mode="r", shape=(2692, 60, 254))

    # Use compact market feature groups to keep SPD stable:
    # take last 20 days and every 8th feature -> about 32 dims.
    feat_idx = np.arange(0, X.shape[2], 8)
    rows = []

    covs = []
    dates = []

    for i in range(len(date_tab)):
        d = pd.to_datetime(date_tab.iloc[i]["Date"])
        seq = np.asarray(X[i, -20:, feat_idx], dtype=np.float64)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        seq = seq - seq.mean(axis=0, keepdims=True)
        cov = np.cov(seq, rowvar=False)
        cov = safe_spd(cov + 1e-3 * np.eye(cov.shape[0]))
        covs.append(cov)
        dates.append(d)

    # reference = log-Euclidean mean on training dates only
    train_logs = []
    for d, cov in zip(dates, covs):
        if d <= pd.Timestamp("2021-12-31"):
            train_logs.append(logm_spd(cov))
    ref = safe_spd(np.linalg.matrix_power(np.eye(covs[0].shape[0]), 1))
    if len(train_logs) > 0:
        ref_log = np.mean(np.stack(train_logs, axis=0), axis=0)
        vals, vecs = np.linalg.eigh(0.5 * (ref_log + ref_log.T))
        ref = (vecs * np.exp(vals)) @ vecs.T
        ref = safe_spd(ref)

    prev_dist = None
    for d, cov in zip(dates, covs):
        dist = affine_invariant_distance(ref, cov)
        rows.append({
            "Date": d,
            "riemann_dist": dist,
            "riemann_dist_chg1": 0.0 if prev_dist is None else dist - prev_dist,
        })
        prev_dist = dist

    out = pd.DataFrame(rows)
    out["riemann_dist_ma5"] = out["riemann_dist"].rolling(5, min_periods=1).mean()
    out["riemann_dist_ma20"] = out["riemann_dist"].rolling(20, min_periods=1).mean()
    out["riemann_dist_z"] = expanding_z(out["riemann_dist"])
    out["riemann_dist_chg1_z"] = expanding_z(out["riemann_dist_chg1"])
    out["riemann_dist_ma5_z"] = expanding_z(out["riemann_dist_ma5"])
    out["riemann_dist_ma20_z"] = expanding_z(out["riemann_dist_ma20"])

    out_path = OUT / "daily_spd_riemannian_market_state.csv"
    out.to_csv(out_path, index=False)
    print("[OK] saved", out_path, out.shape, flush=True)
    return out


def expanding_z(s):
    s = pd.Series(s).astype(float)
    mu = s.expanding().mean().shift(1)
    sd = s.expanding().std().shift(1).replace(0, np.nan)
    z = ((s - mu) / sd).replace([np.inf, -np.inf], 0).fillna(0).clip(-5, 5)
    return z


def add_policy_state(daily, riem=None, use_riem=False):
    d = daily.copy().sort_values("Date").reset_index(drop=True)
    d["Date"] = pd.to_datetime(d["Date"])
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

    if use_riem:
        if riem is None:
            raise ValueError("riem is required when use_riem=True")
        rr = riem.copy()
        rr["Date"] = pd.to_datetime(rr["Date"])
        keep_cols = [
            "Date",
            "riemann_dist_z",
            "riemann_dist_chg1_z",
            "riemann_dist_ma5_z",
            "riemann_dist_ma20_z",
            "riemann_dist",
        ]
        d = d.merge(rr[keep_cols], on="Date", how="left")
        for c in keep_cols:
            if c != "Date" and c in d.columns:
                d[c] = d[c].fillna(0)
        raw_cols += [
            "riemann_dist_z",
            "riemann_dist_chg1_z",
            "riemann_dist_ma5_z",
            "riemann_dist_ma20_z",
        ]

    z_cols = []
    for c in raw_cols:
        if c.endswith("_z"):
            z_cols.append(c)
        else:
            zc = c + "_z"
            d[zc] = expanding_z(d[c])
            z_cols.append(zc)

    return d, z_cols


class SoftmaxPolicy:
    def __init__(self, n_feat, n_act, seed=2022):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.01, size=(n_feat, n_act))
        self.b = np.zeros(n_act, dtype=np.float64)
        self.V = rng.normal(0.0, 0.01, size=(n_feat,))
        self.vb = 0.0

    def logits(self, x):
        return x @ self.W + self.b

    def probs(self, x):
        z = self.logits(x)
        z = z - np.max(z)
        p = np.exp(z)
        return p / np.maximum(p.sum(), 1e-12)

    def value(self, x):
        return float(x @ self.V + self.vb)

    def sample(self, x, rng):
        p = self.probs(x)
        a = int(rng.choice(len(p), p=p))
        return a, p[a], p

    def greedy(self, x):
        return int(np.argmax(self.probs(x)))

    def update_policy(self, x, a, old_prob, adv, lr, clip_eps=0.20, entropy_coef=0.005):
        p = self.probs(x)
        prob = max(float(p[a]), 1e-12)
        ratio = prob / max(old_prob, 1e-12)

        use_grad = True
        if adv > 0 and ratio > 1 + clip_eps:
            use_grad = False
        if adv < 0 and ratio < 1 - clip_eps:
            use_grad = False

        if use_grad:
            grad_logits = -p
            grad_logits[a] += 1.0
            scale = adv * ratio
            self.W += lr * scale * np.outer(x, grad_logits)
            self.b += lr * scale * grad_logits

        # entropy bonus: push away from collapse
        if entropy_coef > 0:
            uniform = np.ones_like(p) / len(p)
            self.W += lr * entropy_coef * np.outer(x, uniform - p)
            self.b += lr * entropy_coef * (uniform - p)

    def update_value(self, x, target, lr):
        pred = self.value(x)
        err = target - pred
        self.V += lr * err * x
        self.vb += lr * err


def rollout_train(daily, state_cols, seed, reward_cfg, epochs=500, lr_policy=0.01, lr_value=0.02):
    rng = np.random.default_rng(seed)
    X = daily[state_cols].to_numpy(dtype=np.float64)
    R = daily["base_ret"].to_numpy(dtype=np.float64)
    pol = SoftmaxPolicy(X.shape[1], len(ACTIONS), seed=seed)

    gamma = reward_cfg.get("gamma", 0.95)
    clip_eps = reward_cfg.get("clip_eps", 0.20)

    for ep in range(epochs):
        traj = []
        prev_gate = 1.0
        eq = 1.0
        peak = 1.0

        for t in range(len(daily)):
            x = X[t]
            a, old_prob, _ = pol.sample(x, rng)
            g = float(ACTIONS[a])
            ret = g * R[t]
            eq *= (1.0 + ret)
            peak = max(peak, eq)
            dd = eq / max(peak, 1e-12) - 1.0
            dd_abs = abs(dd)

            reward = ret
            reward -= reward_cfg["dd_penalty"] * max(0.0, dd_abs - reward_cfg["dd_trigger"])
            reward -= reward_cfg["loss_penalty"] * max(0.0, -ret)
            reward -= reward_cfg["turnover_penalty"] * abs(g - prev_gate)

            traj.append((x, a, old_prob, reward))
            prev_gate = g

        # discounted returns
        G = 0.0
        returns = []
        for _, _, _, rew in reversed(traj):
            G = rew + gamma * G
            returns.append(G)
        returns = list(reversed(returns))
        returns = np.asarray(returns, dtype=np.float64)
        returns_z = (returns - returns.mean()) / (returns.std() + 1e-8)

        for (x, a, old_prob, _), Gz in zip(traj, returns_z):
            v = pol.value(x)
            adv = Gz - v
            pol.update_policy(x, a, old_prob, adv, lr_policy, clip_eps=clip_eps)
            pol.update_value(x, Gz, lr_value)

    return pol


def evaluate_policy(daily, state_cols, pol, name):
    X = daily[state_cols].to_numpy(dtype=np.float64)
    R = daily["base_ret"].to_numpy(dtype=np.float64)

    gates = []
    for t in range(len(daily)):
        a = pol.greedy(X[t])
        gates.append(float(ACTIONS[a]))

    out = daily[["Date", "base_ret"]].copy()
    out["gate"] = gates
    out["ret"] = out["base_ret"] * out["gate"]
    s = summarize_daily(out, "ret", name)
    s["gate_mean"] = float(np.mean(gates))
    s["gate_min"] = float(np.min(gates))
    s["gate_max"] = float(np.max(gates))
    s["turnover_gate"] = float(np.mean(np.abs(np.diff(gates)))) if len(gates) > 1 else 0.0
    s["objective"] = (
        s["ann_sharpe_like"]
        + 0.5 * s["ann_ret_like"]
        + s["positive_day_ratio"]
        + 1.5 * s["max_drawdown_like"]
        - 0.2 * s["turnover_gate"]
    )
    return out, s


def fixed_gate(daily, gate, name):
    out = daily[["Date", "base_ret"]].copy()
    out["gate"] = float(gate)
    out["ret"] = out["base_ret"] * float(gate)
    s = summarize_daily(out, "ret", name)
    s["gate_mean"] = float(gate)
    s["gate_min"] = float(gate)
    s["gate_max"] = float(gate)
    s["turnover_gate"] = 0.0
    s["objective"] = (
        s["ann_sharpe_like"]
        + 0.5 * s["ann_ret_like"]
        + s["positive_day_ratio"]
        + 1.5 * s["max_drawdown_like"]
    )
    return out, s


def worst_best_behavior(base_daily, model_daily, model_name):
    d = base_daily[["Date", "base_ret"]].merge(
        model_daily[["Date", "ret", "gate"]],
        on="Date",
        how="inner"
    )
    worst = d.nsmallest(20, "base_ret")
    best = d.nlargest(20, "base_ret")
    rows = []
    for label, sub in [("worst20_base_days", worst), ("best20_base_days", best)]:
        rows.append({
            "model": model_name,
            "section": label,
            "n_days": int(len(sub)),
            "base_ret_mean": float(sub["base_ret"].mean()),
            "model_ret_mean": float(sub["ret"].mean()),
            "gate_mean": float(sub["gate"].mean()),
            "reduced_exposure_ratio": float((sub["gate"] < 1.0).mean()),
            "full_exposure_ratio": float((sub["gate"] >= 1.0).mean()),
        })
    return pd.DataFrame(rows)


def main():
    print("[BOOT] Riemannian PPO exposure controller", flush=True)

    riem = compute_riemannian_daily_state()

    val0 = read_daily_from_ledger(BASE_LEDGER_VAL)
    test0 = read_daily_from_ledger(BASE_LEDGER)

    val_base, state_cols_base = add_policy_state(val0, riem=None, use_riem=False)
    test_base, _ = add_policy_state(test0, riem=None, use_riem=False)

    val_riem, state_cols_riem = add_policy_state(val0, riem=riem, use_riem=True)
    test_riem, _ = add_policy_state(test0, riem=riem, use_riem=True)

    reward_grid = []
    idx = 0
    for dd_penalty in [0.5, 1.0, 2.0, 4.0]:
        for dd_trigger in [0.05, 0.08, 0.10, 0.15]:
            for loss_penalty in [0.0, 0.2, 0.5]:
                for turnover_penalty in [0.0, 0.01, 0.03]:
                    reward_grid.append({
                        "cfg_idx": idx,
                        "dd_penalty": dd_penalty,
                        "dd_trigger": dd_trigger,
                        "loss_penalty": loss_penalty,
                        "turnover_penalty": turnover_penalty,
                        "gamma": 0.95,
                        "clip_eps": 0.20,
                    })
                    idx += 1

    all_rows = []
    best_pack = {}

    for variant, val, test, state_cols in [
        ("PPO", val_base, test_base, state_cols_base),
        ("Riemannian-PPO", val_riem, test_riem, state_cols_riem),
    ]:
        print(f"[TRAIN] variant={variant} n_state={len(state_cols)}", flush=True)
        best = None
        best_model = None
        best_seed = None
        best_cfg = None
        best_val_daily = None

        for cfg in reward_grid:
            for seed in RNG_SEEDS:
                pol = rollout_train(val, state_cols, seed=seed, reward_cfg=cfg, epochs=400)
                daily_val, sv = evaluate_policy(val, state_cols, pol, f"{variant}_val")
                sv.update({"variant": variant, "split": "val", "seed": seed, **cfg})
                all_rows.append(sv)

                if best is None or sv["objective"] > best["objective"]:
                    best = sv
                    best_model = pol
                    best_seed = seed
                    best_cfg = cfg
                    best_val_daily = daily_val

        daily_test, st = evaluate_policy(test, state_cols, best_model, f"{variant}_test")
        st.update({"variant": variant, "split": "test", "seed": best_seed, **best_cfg})
        all_rows.append(st)

        best_pack[variant] = {
            "policy": best_model,
            "best_val": best,
            "test": st,
            "daily_val": best_val_daily,
            "daily_test": daily_test,
            "state_cols": state_cols,
            "cfg": best_cfg,
            "seed": best_seed,
        }

        daily_test.to_csv(OUT / f"daily_test_{variant.lower().replace('-', '_')}.csv", index=False)
        best_val_daily.to_csv(OUT / f"daily_val_{variant.lower().replace('-', '_')}.csv", index=False)

    # Base and old RL summaries
    base_daily = test0[["Date", "base_ret"]].copy()
    base_daily["ret"] = base_daily["base_ret"]
    base_daily["gate"] = 1.0
    base_summary = summarize_daily(base_daily, "ret", "V2X-B Cross Action Admission")
    base_summary.update({"variant": "Base", "split": "test", "gate_mean": 1.0})

    rows = [base_summary]

    if OLD_RL_TEST.exists():
        old = pd.read_csv(OLD_RL_TEST)
        old["Date"] = pd.to_datetime(old["Date"])
        ret_col = "ret" if "ret" in old.columns else "day_net_ret"
        old_summary = summarize_daily(old, ret_col, "Current PG exposure gate")
        if "gate" in old.columns:
            old_summary["gate_mean"] = float(old["gate"].mean())
        old_summary.update({"variant": "Current-PG", "split": "test"})
        rows.append(old_summary)

    for variant in ["PPO", "Riemannian-PPO"]:
        rows.append(best_pack[variant]["test"])

    final = pd.DataFrame(rows)
    final.to_csv(OUT / "table_riemannian_ppo_main_comparison.csv", index=False)

    search = pd.DataFrame(all_rows)
    search.to_csv(OUT / "valtest_search_riemannian_ppo.csv", index=False)

    # Worst/best behavior
    wb = []
    for variant in ["PPO", "Riemannian-PPO"]:
        wb.append(worst_best_behavior(base_daily, best_pack[variant]["daily_test"], variant))
    if OLD_RL_TEST.exists():
        old = pd.read_csv(OLD_RL_TEST)
        old["Date"] = pd.to_datetime(old["Date"])
        if "ret" in old.columns and "gate" in old.columns:
            wb.append(worst_best_behavior(base_daily, old[["Date", "ret", "gate"]], "Current-PG"))
    wb = pd.concat(wb, ignore_index=True)
    wb.to_csv(OUT / "table_riemannian_ppo_worst_best_behavior.csv", index=False)

    report = {
        "method": "Risk-sensitive PPO exposure controller with optional SPD-Riemannian market-state features.",
        "actions": ACTIONS.tolist(),
        "base_ledger_test": str(BASE_LEDGER),
        "base_ledger_val": str(BASE_LEDGER_VAL),
        "best": {
            k: {
                "seed": v["seed"],
                "cfg": v["cfg"],
                "state_cols": v["state_cols"],
                "test": v["test"],
            }
            for k, v in best_pack.items()
        }
    }
    (OUT / "report_riemannian_ppo_exposure.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n===== RIEMANNIAN PPO MAIN COMPARISON =====")
    print(final.to_string(index=False))

    print("\n===== WORST BEST BEHAVIOR =====")
    print(wb.to_string(index=False))

    print("\n[OK] saved to", OUT)


if __name__ == "__main__":
    main()
