"""
Dataset Loading and Preprocessing Script
=========================================
Grid-Tied Solar Inverter Multi-Class Fault Detection Dataset
5-Class: Normal, SLG(A-G), SLG(B-G), SLG(C-G), Three-Phase Short

Paper: Patel et al. (2026), IEEE GreenTech Conference
DOI: 10.1109/GreenTech68285.2026.11471570

Usage:
    python scripts/load_dataset.py --data_path data/inverter_fault_dataset.csv
    python scripts/load_dataset.py --save_npy  # Save pre-processed numpy arrays
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ─── Constants ───────────────────────────────────────────────────────────────
FEATURE_COLUMNS = ['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic', 'Id', 'Iq', 'P', 'Q', 'THD']
LABEL_COLUMN = 'label'
NUM_CLASSES = 5
WINDOW_SIZE = 100      # 100 samples = 10 ms at 10 kHz
STEP_SIZE = 20         # 20 samples = 2 ms
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

CLASS_NAMES = {
    0: 'Normal Operation',
    1: 'SLG Fault (A-G)',
    2: 'SLG Fault (B-G)',
    3: 'SLG Fault (C-G)',
    4: 'Three-Phase Short (A-B-C)',
}


def load_raw_data(csv_path: str) -> tuple:
    """Load raw CSV and return features and labels."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    X = df[FEATURE_COLUMNS].values   # (N, 11)
    y = df[LABEL_COLUMN].values      # (N,)

    # Validate 5-class labels
    unique_labels = np.unique(y)
    assert all(l in range(NUM_CLASSES) for l in unique_labels), \
        f"Expected labels 0-{NUM_CLASSES-1}, got {unique_labels}"

    print(f"\nClass distribution:")
    for label in sorted(unique_labels):
        count = np.sum(y == label)
        print(f"  Label {label} ({CLASS_NAMES[label]}): {count:,} samples")

    return X, y


def normalize_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Z-score normalization. Fit on training split ONLY to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_val_norm, X_test_norm, scaler


def create_sliding_windows(X: np.ndarray, y: np.ndarray,
                           window_size: int = WINDOW_SIZE,
                           step_size: int = STEP_SIZE) -> tuple:
    """
    Create sliding window sequences with majority-vote labeling.

    Parameters
    ----------
    X : array of shape (N, 11), normalized features
    y : array of shape (N,), integer labels
    window_size : int, number of samples per window (default 100 = 10 ms)
    step_size : int, stride between windows (default 20 = 2 ms)

    Returns
    -------
    X_seq : array of shape (N_windows, window_size, 11)
    y_seq : array of shape (N_windows,)
    """
    sequences, labels = [], []
    for i in range(0, len(X) - window_size, step_size):
        sequences.append(X[i:i + window_size])
        # Majority vote for window label
        window_labels = y[i:i + window_size].astype(int)
        labels.append(np.bincount(window_labels, minlength=NUM_CLASSES).argmax())

    X_seq = np.array(sequences, dtype=np.float32)
    y_seq = np.array(labels, dtype=np.int32)

    print(f"Created {X_seq.shape[0]} windows: shape {X_seq.shape}")
    return X_seq, y_seq


def prepare_dataset(csv_path: str, save_npy: bool = False, output_dir: str = 'data'):
    """Full pipeline: load → split → normalize → window → (optionally save)."""

    # Step 1: Load raw data
    X, y = load_raw_data(csv_path)

    # Step 2: Stratified train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED, stratify=y
    )
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test,
        random_state=RANDOM_SEED, stratify=y_temp
    )
    print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Step 3: Z-score normalization (fit on train only)
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(
        X_train, X_val, X_test
    )

    # Step 4: Create sliding windows
    print("\n--- Training windows ---")
    X_train_seq, y_train_seq = create_sliding_windows(X_train_norm, y_train)
    print("--- Validation windows ---")
    X_val_seq, y_val_seq = create_sliding_windows(X_val_norm, y_val)
    print("--- Test windows ---")
    X_test_seq, y_test_seq = create_sliding_windows(X_test_norm, y_test)

    # Step 5: Save if requested
    if save_npy:
        import os
        os.makedirs(output_dir, exist_ok=True)
        np.save(f'{output_dir}/X_train.npy', X_train_seq)
        np.save(f'{output_dir}/y_train.npy', y_train_seq)
        np.save(f'{output_dir}/X_val.npy', X_val_seq)
        np.save(f'{output_dir}/y_val.npy', y_val_seq)
        np.save(f'{output_dir}/X_test.npy', X_test_seq)
        np.save(f'{output_dir}/y_test.npy', y_test_seq)
        print(f"\nSaved .npy files to {output_dir}/")

    return {
        'X_train': X_train_seq, 'y_train': y_train_seq,
        'X_val': X_val_seq, 'y_val': y_val_seq,
        'X_test': X_test_seq, 'y_test': y_test_seq,
        'scaler': scaler, 'class_names': CLASS_NAMES,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and preprocess inverter fault dataset')
    parser.add_argument('--data_path', type=str, default='data/inverter_fault_dataset.csv',
                        help='Path to raw CSV file')
    parser.add_argument('--save_npy', action='store_true',
                        help='Save pre-processed numpy arrays')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for .npy files')
    args = parser.parse_args()

    data = prepare_dataset(args.data_path, save_npy=args.save_npy, output_dir=args.output_dir)
    print(f"\n✓ Dataset ready: {data['X_train'].shape[0]} train, "
          f"{data['X_val'].shape[0]} val, {data['X_test'].shape[0]} test windows")
