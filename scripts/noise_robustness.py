"""
Noise Robustness Analysis
==========================
Evaluate model accuracy under varying Gaussian noise levels (0–15%).
Reproduces the noise robustness results from the paper.

Usage:
    python scripts/noise_robustness.py
    python scripts/noise_robustness.py --noise_levels 0 2.5 5 7.5 10 12.5 15
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score


NUM_CLASSES = 5
CLASS_NAMES = [
    'Normal', 'SLG (A-G)', 'SLG (B-G)', 'SLG (C-G)', '3-Phase Short'
]


def add_gaussian_noise(X: np.ndarray, noise_percent: float) -> np.ndarray:
    """
    Add Gaussian noise to normalized data.

    Parameters
    ----------
    X : array, input data (already Z-score normalized)
    noise_percent : float, noise level as percentage (e.g., 5.0 for 5%)

    Returns
    -------
    X_noisy : array, data with added noise
    """
    if noise_percent == 0:
        return X.copy()
    sigma = noise_percent / 100.0
    noise = np.random.normal(0, sigma, X.shape).astype(np.float32)
    return X + noise


def run_noise_analysis(args):
    """Evaluate model under multiple noise levels."""

    # Load data
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))

    # Load model
    model = tf.keras.models.load_model(args.model_path)
    print(f"Model: {args.model_path}")
    print(f"Test set: {X_test.shape[0]} windows\n")

    noise_levels = [float(n) for n in args.noise_levels]
    results = []

    print(f"{'Noise Level':>12s} | {'Accuracy':>10s}")
    print('-' * 28)

    for noise_pct in noise_levels:
        X_noisy = add_gaussian_noise(X_test, noise_pct)
        y_pred = np.argmax(model.predict(X_noisy, batch_size=256, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred) * 100
        results.append((noise_pct, acc))
        marker = ' ← training noise' if noise_pct == 5.0 else ''
        print(f"{noise_pct:>11.1f}% | {acc:>9.2f}%{marker}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    noise_vals, acc_vals = zip(*results)
    ax.plot(noise_vals, acc_vals, 'o-', color='#2196F3', linewidth=2, markersize=8)
    ax.axvline(x=5.0, color='red', linestyle='--', alpha=0.7, label='Training noise (5%)')
    ax.set_xlabel('Gaussian Noise Level (%)', fontsize=12)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_title('CNN-LSTM Noise Robustness Analysis', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0, min(acc_vals) - 5), top=101)
    plt.tight_layout()

    if args.save_figure:
        os.makedirs('figures', exist_ok=True)
        fig.savefig('figures/noise_robustness.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: figures/noise_robustness.png")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Noise robustness analysis')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_path', type=str, default='models/cnn_lstm_best.h5')
    parser.add_argument('--noise_levels', nargs='+', default=[0, 2.5, 5, 7.5, 10, 12.5, 15])
    parser.add_argument('--save_figure', action='store_true')
    args = parser.parse_args()

    run_noise_analysis(args)
