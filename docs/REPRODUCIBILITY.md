# Reproducibility Notes

## CAMP Pipeline

CAMP follows a role-separated pipeline:

1. train evidence experts;
2. build same-day cross-sectional admission;
3. project admitted stocks into a base Top-K portfolio;
4. apply SPD-state exposure protection;
5. evaluate under a frozen chronological test protocol.

## Protocol Rules

- Normalization statistics are fitted only on the training split.
- Admission hyperparameters are selected only on the validation/calibration split.
- Exposure-controller hyperparameters are selected only on validation/calibration data.
- The frozen test split is used only for final reporting.
- Exposure gates rescale the admitted portfolio and do not reselect stocks.
- Transaction friction is charged by turnover.

## Sample Usage

Run `python examples/inspect_sample.py data/sample_1pct 0` to inspect the first sample shard.

The 1% sample is designed for format verification and code inspection. It should not be used to reproduce full-scale performance metrics.
