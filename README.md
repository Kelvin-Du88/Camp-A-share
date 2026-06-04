# CAMP-A-share

CAMP is a research framework for **Cross-sectional Admission with Market-panel Protection** in non-stationary China A-share market panels.

The core idea is to treat each trading day as a structured market evidence field rather than a set of independent stock-date samples. CAMP separates temporal evidence mining, same-day cross-sectional admission, portfolio projection, and SPD-state exposure protection into an auditable evidence-to-decision pipeline.

## Main Components

- **Temporal evidence mining**: extracts recall-oriented candidate signals from 60-day stock sequences.
- **Cross-sectional admission**: standardizes heterogeneous evidence within each trading day and admits candidates into the daily Top-K portfolio.
- **Market-body evidence**: incorporates market-wide breadth, liquidity, limit-state, and index-movement information.
- **Portfolio projection**: converts admitted scores into long-only portfolio weights under exposure and concentration constraints.
- **SPD-state exposure gate**: rescales aggregate exposure using covariance-geometry market states without reselecting stocks.

## Repository Layout

- `src/experts/`: temporal, cross-sectional, and market-body evidence modules.
- `src/decision/`: admission, portfolio projection, exposure gates, and geometry overlays.
- `src/evaluation/`: RankIC diagnostics, execution stress tests, and failure analysis.
- `src/data/`: 1% sample dataset loader.
- `data/sample_1pct/`: processed 1% sample dataset.
- `docs/`: dataset and reproducibility notes.
- `examples/`: small loading examples.

## Quick Start

Install dependencies with `pip install -r requirements.txt`.

Inspect the sample dataset with `python examples/inspect_sample.py data/sample_1pct 0`.

## Dataset

This repository includes a reproducible 1% processed sample dataset for code inspection and pipeline demonstration. The full processed dataset is not included in the public repository.

The released sample contains 94,766 sampled stock-date rows, stock-level 60-day sequence features, same-day cross-sectional features, market-level 60-day sequence features, 5-day return targets, and metadata with chronological split labels.

More details are available in `docs/DATASET.md`.

## Research Protocol Note

All reported returns are return-like ledger metrics under a fixed research protocol. They are used for relative research comparison only and should not be interpreted as live trading claims.

## Citation

If you use this repository, please cite the corresponding CAMP paper once available.

## License

This repository is released for academic research and reproducibility. See `LICENSE`.
