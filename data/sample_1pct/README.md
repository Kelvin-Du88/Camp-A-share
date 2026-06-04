# CAMP 1% Processed Sample Dataset

This folder contains a reproducible 1% processed sample of the CAMP A-share market-panel dataset.

## Files

- `sample_meta_1pct.parquet`: metadata for sampled stock-date rows.
- `sample_shard_*.npz`: row-wise feature shards.
- `sample_shard_index.csv`: shard index.
- `X_market_seq_by_date_1pct.npy`: market-level 60-day sequence subset used by sampled rows.
- `market_seq_index_map_1pct.csv`: mapping from sample market index to original market index.
- `feature_info_L60_H5_full.json`: feature column definitions.
- `period_split_info_L60_H5_full.json`: chronological split information.
- `normalization_stats_L60_H5_full.json`: training-split normalization statistics.

## NPZ shard keys

Each `sample_shard_*.npz` contains:

- `X_stock_seq`: stock-level 60-day sequence features.
- `X_cross_today`: same-day cross-sectional features.
- `y_ret_5d`: future 5-day return target.
- `y_up_5d`: future 5-day up/down target.
- `market_seq_idx`: remapped row index into `X_market_seq_by_date_1pct.npy`.
- `original_row_index`: row index in the full processed dataset.

## Protocol Note

This sample is released only for reproducibility, code inspection, and research demonstration.  
All reported returns in the paper are return-like ledger metrics under a fixed research protocol and are not live trading claims.
