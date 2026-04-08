# Data Directory

## File Descriptions

| File | Format | Description |
|------|--------|-------------|
| `inverter_fault_dataset.csv` | CSV | Raw time-series data with 13 columns (time, 11 features, label) |
| `X_train.npy` | NumPy | Training features, shape `(N_train, 100, 11)` |
| `y_train.npy` | NumPy | Training labels, shape `(N_train,)` |
| `X_val.npy` | NumPy | Validation features, shape `(N_val, 100, 11)` |
| `y_val.npy` | NumPy | Validation labels, shape `(N_val,)` |
| `X_test.npy` | NumPy | Test features, shape `(N_test, 100, 11)` |
| `y_test.npy` | NumPy | Test labels, shape `(N_test,)` |

## CSV Column Format

```
time, Va, Vb, Vc, Ia, Ib, Ic, Id, Iq, P, Q, THD, label
```

- Columns 1–11: Electrical features (see main README for full descriptions)
- Column 12 (`label`): Integer fault class (0–4)

## Label Mapping

| Label | Fault Type |
|:-----:|------------|
| 0 | Normal Operation |
| 1 | Single Line-to-Ground Fault (A–G) |
| 2 | Single Line-to-Ground Fault (B–G) |
| 3 | Single Line-to-Ground Fault (C–G) |
| 4 | Three-Phase Short Circuit (A–B–C) |

## Data Split

- **Training**: 70% (stratified)
- **Validation**: 15% (stratified)
- **Testing**: 15% (stratified)

## Preprocessing Applied to .npy Files

1. Z-score normalization (μ, σ computed from training split only)
2. Sliding window: W=100 samples (10 ms at 10 kHz), step=20 samples
3. Majority-vote labeling per window

## Generating .npy Files from CSV

```bash
python scripts/load_dataset.py --save_npy --data_path data/inverter_fault_dataset.csv
```

## Note on Large Files

The CSV and .npy files may be hosted on [Zenodo](https://zenodo.org/records/19463598) due to GitHub file size limits. Download them and place in this directory before running training scripts.
