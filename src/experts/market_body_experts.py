#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V2M/V2N Financial Deep Experts.

V2M: MASTER-style market-guided temporal expert.
     stock_seq + cross_today + market_state -> score

V2N: StockMixer-style temporal/channel mixer expert.
     stock_seq + cross_today -> score

Outputs:
  summary_<model>_expert.csv
  score_val/test_<model>.csv
  daily_val/test_<model>.csv
  ledger_val/test_<model>.csv
  final_<model>_report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    if len(raw) == 0:
        return raw
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


class MemmapDataset(Dataset):
    def __init__(self, idx, X_stock, X_cross, y, market_idx=None, market_feat=None, use_market=False, preload=False):
        self.idx = np.asarray(idx, dtype=np.int64)
        self.X_stock = X_stock
        self.X_cross = X_cross
        self.y = y
        self.market_idx = market_idx
        self.market_feat = market_feat
        self.use_market = use_market
        self.preload = preload

        if preload:
            print(f"[PRELOAD] rows={len(self.idx)} use_market={use_market}", flush=True)
            self.stock_cache = np.asarray(X_stock[self.idx], dtype=np.float32).copy()
            self.cross_cache = np.asarray(X_cross[self.idx], dtype=np.float32).copy()
            self.y_cache = np.asarray(y[self.idx], dtype=np.float32).copy()
            if use_market:
                mids = np.asarray(market_idx[self.idx], dtype=np.int64)
                self.market_cache = np.asarray(market_feat[mids], dtype=np.float32).copy()
            else:
                self.market_cache = None
        else:
            self.stock_cache = None
            self.cross_cache = None
            self.y_cache = None
            self.market_cache = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        if self.preload:
            xs = self.stock_cache[j]
            xc = self.cross_cache[j]
            yy = np.float32(self.y_cache[j])
            if self.use_market:
                xm = self.market_cache[j]
                return torch.from_numpy(xs.copy()), torch.from_numpy(xc.copy()), torch.from_numpy(xm.copy()), torch.tensor(yy)
            return torch.from_numpy(xs.copy()), torch.from_numpy(xc.copy()), torch.tensor(yy)

        i = int(self.idx[j])
        xs = np.asarray(self.X_stock[i], dtype=np.float32)
        xc = np.asarray(self.X_cross[i], dtype=np.float32)
        yy = np.float32(self.y[i])

        if self.use_market:
            mi = int(self.market_idx[i])
            xm = np.asarray(self.market_feat[mi], dtype=np.float32)
            return torch.from_numpy(xs.copy()), torch.from_numpy(xc.copy()), torch.from_numpy(xm.copy()), torch.tensor(yy)
        return torch.from_numpy(xs.copy()), torch.from_numpy(xc.copy()), torch.tensor(yy)


class MasterStyleExpert(nn.Module):
    """
    MASTER-style:
    - market state produces a feature gate
    - gated stock sequence goes through temporal attention
    - cross-sectional today features are fused into prediction
    """
    def __init__(self, stock_dim, cross_dim, market_dim, hidden=128, dropout=0.15):
        super().__init__()
        self.market_encoder = nn.Sequential(
            nn.Linear(market_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, stock_dim),
            nn.Sigmoid(),
        )
        self.stock_proj = nn.Linear(stock_dim, hidden)
        self.temporal_score = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        self.cross_proj = nn.Sequential(
            nn.Linear(cross_dim, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden + hidden // 2),
            nn.Linear(hidden + hidden // 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, xs, xc, xm):
        # xm: [B, market_dim], precomputed concat(last, mean)
        gate = self.market_encoder(xm)  # [B,F]
        xs = xs * gate[:, None, :]

        h = self.stock_proj(xs)  # [B,L,H]
        a = self.temporal_score(h).squeeze(-1)
        a = torch.softmax(a, dim=1)
        token = (h * a[:, :, None]).sum(dim=1)

        c = self.cross_proj(xc)
        out = self.head(torch.cat([token, c], dim=-1)).squeeze(-1)
        return out


class StockMixerStyleExpert(nn.Module):
    """
    StockMixer-style lightweight:
    - time mixing over 60-day sequence
    - channel mixing over stock features
    - cross_today fusion
    Note: stock-to-stock mixing is represented at portfolio construction and later router level.
    """
    def __init__(self, seq_len, stock_dim, cross_dim, hidden=128, dropout=0.15):
        super().__init__()
        self.time_mixer = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.channel_mixer = nn.Sequential(
            nn.Linear(stock_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.cross_proj = nn.Sequential(
            nn.Linear(cross_dim, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden + hidden // 2),
            nn.Linear(hidden + hidden // 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, xs, xc):
        # xs [B,L,F] -> transpose [B,F,L], mix time per channel
        xt = xs.transpose(1, 2)
        t = self.time_mixer(xt).squeeze(-1)  # [B,F]
        h = self.channel_mixer(t)
        c = self.cross_proj(xc)
        return self.head(torch.cat([h, c], dim=-1)).squeeze(-1)


def train_model(args, model, train_loader, val_loader, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_state = None
    best_val = float("inf")
    bad = 0

    loss_fn = nn.SmoothL1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                if args.model == "master":
                    xs, xc, xm, y = batch
                    pred = model(xs.to(device, non_blocking=True), xc.to(device, non_blocking=True), xm.to(device, non_blocking=True))
                else:
                    xs, xc, y = batch
                    pred = model(xs.to(device, non_blocking=True), xc.to(device, non_blocking=True))

                y = y.to(device, non_blocking=True)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tr_losses.append(float(loss.detach().cpu()))

        model.eval()
        va_losses = []
        with torch.no_grad():
            for batch in val_loader:
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    if args.model == "master":
                        xs, xc, xm, y = batch
                        pred = model(xs.to(device, non_blocking=True), xc.to(device, non_blocking=True), xm.to(device, non_blocking=True))
                    else:
                        xs, xc, y = batch
                        pred = model(xs.to(device, non_blocking=True), xc.to(device, non_blocking=True))
                    loss = loss_fn(pred, y.to(device, non_blocking=True))
                va_losses.append(float(loss.detach().cpu()))

        tr = float(np.mean(tr_losses))
        va = float(np.mean(va_losses))
        print(f"[EPOCH] {epoch:03d} train_loss={tr:.6f} val_loss={va:.6f}", flush=True)

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[EARLY_STOP] epoch={epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def predict_scores(args, model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                if args.model == "master":
                    xs, xc, xm, y = batch
                    pred = model(xs.to(device, non_blocking=True), xc.to(device, non_blocking=True), xm.to(device, non_blocking=True))
                else:
                    xs, xc, y = batch
                    pred = model(xs.to(device, non_blocking=True), xc.to(device, non_blocking=True))
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds)


def backtest_scores(meta_split, idx, score, y_decimal, args):
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
    ap.add_argument("--model", choices=["master", "stockmixer"], required=True)
    ap.add_argument("--data_dir", default="data_external/msan_samples_full_qfq_norm")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--max_train_rows", type=int, default=600000)
    ap.add_argument("--max_val_fit_rows", type=int, default=160000)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--pred_batch_size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--random_seed", type=int, default=42)

    ap.add_argument("--horizon_days", type=float, default=5.0)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--exposure", type=float, default=0.5)
    ap.add_argument("--max_weight", type=float, default=0.10)
    ap.add_argument("--temp", type=float, default=0.35)

    args = ap.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BOOT] model={args.model} device={device}", flush=True)

    feature_info = read_json(data_dir / "feature_info_L60_H5_full.json")
    meta = normalize_meta(pd.read_parquet(data_dir / "meta_L60_H5_full.parquet"))
    n = len(meta)

    x_stock_shape = feature_info.get("X_stock_seq_shape") or [n, 60, 26]
    x_cross_shape = feature_info.get("X_cross_today_shape") or [n, 13]

    X_stock = np.memmap(data_dir / "X_stock_seq_L60_H5_full.dat", dtype="float32", mode="r", shape=tuple(x_stock_shape))
    X_cross = np.memmap(data_dir / "X_cross_today_L60_H5_full.dat", dtype="float32", mode="r", shape=tuple(x_cross_shape))
    y = np.asarray(np.memmap(data_dir / "y_ret_5d_L60_H5_full.dat", dtype="float32", mode="r", shape=(n,)), dtype=np.float32) / 100.0

    use_market = args.model == "master"
    X_market = None
    market_idx = None
    market_feat = None
    if use_market:
        market_path_npy = data_dir / "X_market_seq_by_date_L60_H5_full.npy"
        X_market = np.load(market_path_npy, mmap_mode="r")
        idx_path = data_dir / "market_seq_idx_L60_H5_full.dat"
        idx_bytes = idx_path.stat().st_size
        if idx_bytes == n * 8:
            idx_dtype = "int64"
        elif idx_bytes == n * 4:
            idx_dtype = "int32"
        else:
            raise ValueError(f"Unexpected market_seq_idx file size: bytes={idx_bytes}, n={n}")
        print(f"[LOAD] market_seq_idx dtype={idx_dtype} bytes={idx_bytes}", flush=True)
        market_idx = np.memmap(idx_path, dtype=idx_dtype, mode="r", shape=(n,))
        print(f"[LOAD] market={X_market.shape}", flush=True)
        print("[BUILD] cached market features concat(last, mean)", flush=True)
        market_last = np.asarray(X_market[:, -1, :], dtype=np.float32)
        market_mean = np.asarray(X_market.mean(axis=1), dtype=np.float32)
        market_feat = np.concatenate([market_last, market_mean], axis=1).astype(np.float32)
        print(f"[BUILD] market_feat={market_feat.shape}", flush=True)

    train_idx = split_idx(meta, "train")
    val_idx = split_idx(meta, "val")
    test_idx = split_idx(meta, "test")

    rng = np.random.default_rng(args.random_seed)
    if len(train_idx) > args.max_train_rows:
        train_fit_idx = np.sort(rng.choice(train_idx, size=args.max_train_rows, replace=False))
    else:
        train_fit_idx = np.sort(train_idx)

    if len(val_idx) > args.max_val_fit_rows:
        val_fit_idx = np.sort(rng.choice(val_idx, size=args.max_val_fit_rows, replace=False))
    else:
        val_fit_idx = np.sort(val_idx)

    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)

    print(f"[DATA] train_fit={len(train_fit_idx)} val_fit={len(val_fit_idx)} val_full={len(val_idx)} test_full={len(test_idx)}", flush=True)

    train_ds = MemmapDataset(train_fit_idx, X_stock, X_cross, y, market_idx, market_feat, use_market, preload=True)
    val_fit_ds = MemmapDataset(val_fit_idx, X_stock, X_cross, y, market_idx, market_feat, use_market, preload=True)
    val_full_ds = MemmapDataset(val_idx, X_stock, X_cross, y, market_idx, market_feat, use_market, preload=False)
    test_full_ds = MemmapDataset(test_idx, X_stock, X_cross, y, market_idx, market_feat, use_market, preload=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_fit_ds, batch_size=args.pred_batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_full_loader = DataLoader(val_full_ds, batch_size=args.pred_batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_full_loader = DataLoader(test_full_ds, batch_size=args.pred_batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    seq_len = int(x_stock_shape[1])
    stock_dim = int(x_stock_shape[2])
    cross_dim = int(x_cross_shape[1])

    if args.model == "master":
        market_dim = int(market_feat.shape[1])
        model = MasterStyleExpert(stock_dim, cross_dim, market_dim, hidden=args.hidden, dropout=args.dropout)
        strategy = "v2m_master_style_expert"
        prefix = "v2m_master_style"
    else:
        model = StockMixerStyleExpert(seq_len, stock_dim, cross_dim, hidden=args.hidden, dropout=args.dropout)
        strategy = "v2n_stockmixer_style_expert"
        prefix = "v2n_stockmixer_style"

    model.to(device)

    print("[TRAIN] start", flush=True)
    best_val_loss = train_model(args, model, train_loader, val_loader, device)
    torch.save(model.state_dict(), out_dir / f"{prefix}.pt")

    print("[PRED] val full", flush=True)
    pred_val = predict_scores(args, model, val_full_loader, device)
    print("[PRED] test full", flush=True)
    pred_test = predict_scores(args, model, test_full_loader, device)

    val_meta = meta.iloc[val_idx][["Date", "Ticker"]].copy().reset_index(drop=True)
    test_meta = meta.iloc[test_idx][["Date", "Ticker"]].copy().reset_index(drop=True)

    daily_val, ledger_val, score_val = backtest_scores(val_meta, val_idx, pred_val, y, args)
    daily_test, ledger_test, score_test = backtest_scores(test_meta, test_idx, pred_test, y, args)

    score_val.to_csv(out_dir / f"score_val_{prefix}.csv", index=False)
    score_test.to_csv(out_dir / f"score_test_{prefix}.csv", index=False)
    daily_val.to_csv(out_dir / f"daily_val_{prefix}.csv", index=False)
    daily_test.to_csv(out_dir / f"daily_test_{prefix}.csv", index=False)
    ledger_val.to_csv(out_dir / f"ledger_val_{prefix}.csv", index=False)
    ledger_test.to_csv(out_dir / f"ledger_test_{prefix}.csv", index=False)

    summary = pd.DataFrame([
        {**summarize(daily_val, strategy, args.horizon_days), "split": "val"},
        {**summarize(daily_test, strategy, args.horizon_days), "split": "test"},
    ])
    summary.to_csv(out_dir / f"summary_{prefix}_expert.csv", index=False)

    if args.model == "master":
        method = "V2M MASTER-style Market-Guided Financial Transformer Expert"
        formula = "Market state generates feature gate; temporal attention aggregates stock sequence; cross_today features are fused into score."
        family = "financial_transformer"
    else:
        method = "V2N StockMixer-style Cross-sectional/Temporal Mixer Expert"
        formula = "Time mixing over stock sequence plus channel mixing and cross_today fusion; portfolio-level TopK realizes cross-sectional selection."
        family = "financial_mlp_mixer"

    report = {
        "method": method,
        "family": family,
        "formula": formula,
        "selection_protocol": "Train on train split; monitor validation loss; apply frozen expert to validation/test backtest.",
        "best_val_loss": float(best_val_loss),
        "config": vars(args),
        "val": summary.iloc[0].to_dict(),
        "test": summary.iloc[1].to_dict(),
        "outputs": [
            f"summary_{prefix}_expert.csv",
            f"score_val_{prefix}.csv",
            f"score_test_{prefix}.csv",
            f"daily_val_{prefix}.csv",
            f"daily_test_{prefix}.csv",
            f"ledger_val_{prefix}.csv",
            f"ledger_test_{prefix}.csv",
            f"{prefix}.pt",
        ],
    }
    (out_dir / f"final_{prefix}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== SUMMARY =====")
    print(summary.to_string(index=False))
    print(f"[OK] saved: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
