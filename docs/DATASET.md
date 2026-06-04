# Dataset Description

## Overview

The full CAMP processed dataset contains 9,476,695 China A-share stock-date samples. Each sample is built from a 60-day lookback window and a 5-day prediction horizon.

The public repository includes a 1% processed sample for code inspection and reproducibility demonstration.

## Released 1% Sample

The sample is stored under `data/sample_1pct/`.

It contains:

- 94,766 sampled stock-date rows;
- stock-level sequence features with shape 60 × 26 per row;
- same-day cross-sectional features with 13 dimensions per row;
- market-level sequence features with shape 60 × 254 per market date;
- future 5-day return and up/down targets;
- metadata, feature definitions, row shards, and market sequence shards.

## Main Files

- `sample_manifest_1pct.json`: sample dataset manifest.
- `sample_meta_1pct.parquet`: sampled metadata.
- `sample_shard_index.csv`: index of row-level shards.
- `sample_shard_*.npz`: row-level data shards.
- `market_seq_shards/`: sharded market sequence arrays.
- `market_seq_shard_index.csv`: index of market sequence shards.
- `market_seq_index_map_1pct.csv`: mapping between sampled rows and market sequence rows.
- `feature_info_L60_H5_full.json`: feature column definitions.

## Row Shard Fields

Each `sample_shard_*.npz` file contains:

- `X_stock_seq`
- `X_cross_today`
- `y_ret_5d`
- `y_up_5d`
- `market_seq_idx`
- `original_row_index`

The corresponding market sequence can be loaded through `src/data/sample_loader.py`.

## Chronological Protocol

The full research protocol uses chronological splits:

- train: 2016-01-04 to 2021-12-31;
- validation/calibration: 2022-01-04 to 2023-12-29;
- frozen test: 2024-01-02 to 2026-04-23.

The sample preserves split metadata when available.

## Limitation

The 1% sample is not intended to reproduce the full-paper performance. It is provided to verify data format, code structure, and the CAMP evidence-to-decision pipeline.
