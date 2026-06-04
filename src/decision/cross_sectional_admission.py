import argparse
import numpy as np
import pandas as pd
from v2x_common import load_panel, zscore_by_date, backtest, save_result, ensure_dir


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_external/msan_samples_full_qfq_norm")
    ap.add_argument("--out_dir", default="outputs/dive_trader_v2/v2x_b_cross_action_admission")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--candidate_m", type=int, default=120)
    ap.add_argument("--exposure", type=float, default=0.5)
    ap.add_argument("--max_weight", type=float, default=0.10)
    ap.add_argument("--temp", type=float, default=0.10)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    return ap.parse_args()


def train_linear_ranker(df, feature_cols):
    # pair-free simple rank regression on validation candidate pool
    x = df[feature_cols].to_numpy(dtype="float64")
    y = df["future_return_5d"].to_numpy(dtype="float64")
    y = (y - y.mean()) / (y.std() + 1e-9)
    x = np.nan_to_num(x)
    w = np.zeros(x.shape[1])
    lr = 0.03
    for _ in range(500):
        pred = x @ w
        grad = x.T @ (pred - y) / max(len(y), 1) + 1e-3 * w
        w -= lr * grad
    return w


def build_candidate_pool(panel, m):
    return panel.sort_values(["Date", "temporal_z"], ascending=[True, False]).groupby("Date", group_keys=False).head(m).copy()


def main():
    args = parse_args()
    print("[BOOT] 61b Cross Action Admission", flush=True)

    score_paths = {
        "temporal": "outputs/dive_trader_v2/experts/v2l_frequency_expert/score_{split}_v2l_frequency_expert.csv",
        "cross": "outputs/dive_trader_v2/experts/factor_tree/score_{split}_factor_tree.csv",
        "body_m": "outputs/dive_trader_v2/experts/v2m_master_style/score_{split}_v2m_master_style.csv",
        "body_n": "outputs/dive_trader_v2/experts/v2n_stockmixer_style/score_{split}_v2n_stockmixer_style.csv",
        "rankic": "outputs/dive_trader_v2/experts/v2j_rankic_linear/score_{split}_v2j_rankic_linear.csv",
    }

    panels = {}
    for split in ["val", "test"]:
        paths = {k: v.format(split=split) for k, v in score_paths.items()}
        p = load_panel(args.data_dir, split, paths)
        p = zscore_by_date(p, ["temporal", "cross", "body_m", "body_n", "rankic"])
        panels[split] = p

    rows = []
    best = None
    best_pack = None

    for m in [80, 120, 160, 200, 300]:
        train_pool = build_candidate_pool(panels["val"], m)
        feature_cols = ["cross_z", "body_m_z", "body_n_z", "rankic_z", "temporal_z"]
        w = train_linear_ranker(train_pool, feature_cols)

        for wt in [0.5, 1.0, 1.5]:
            p = build_candidate_pool(panels["val"], m)
            p["action_score"] = p[feature_cols].to_numpy(dtype="float64") @ w
            p["score_final"] = p["temporal_z"] + wt * p["action_score"]

            _, _, sv = backtest(
                p, "score_final",
                topk=args.topk, exposure=args.exposure, max_weight=args.max_weight,
                temp=args.temp, cost_bps=args.cost_bps
            )
            sv.update({"split": "val", "candidate_m": m, "action_weight": wt})
            rows.append(sv)
            if best is None or sv["objective"] > best["objective"]:
                best = sv
                best_pack = (m, wt, w.tolist(), feature_cols)

    m, wt, w, feature_cols = best_pack
    summaries = []
    ledgers = []
    dailies = []

    for split in ["val", "test"]:
        p = build_candidate_pool(panels[split], m)
        p["action_score"] = p[feature_cols].to_numpy(dtype="float64") @ np.asarray(w)
        p["score_final"] = p["temporal_z"] + wt * p["action_score"]
        ledger, daily, s = backtest(
            p, "score_final",
            topk=args.topk, exposure=args.exposure, max_weight=args.max_weight,
            temp=args.temp, cost_bps=args.cost_bps
        )
        s.update({"strategy": "v2x_b_cross_action_admission", "split": split, "candidate_m": m, "action_weight": wt})
        summaries.append(s)
        ledgers.append(ledger.assign(split=split))
        dailies.append(daily.assign(split=split))

    out = ensure_dir(args.out_dir)
    pd.DataFrame(rows).sort_values("objective", ascending=False).to_csv(out / "val_search_61b.csv", index=False)
    save_result(
        out, "61b_cross_action_admission",
        pd.concat(ledgers, ignore_index=True),
        pd.concat(dailies, ignore_index=True),
        summaries,
        extra={"method": "V2X-b Cross-sectional Action Admission", "best": {"candidate_m": m, "action_weight": wt}, "features": feature_cols, "weights": w}
    )


if __name__ == "__main__":
    main()
