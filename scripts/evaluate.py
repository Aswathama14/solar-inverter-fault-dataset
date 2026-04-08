"""
Evaluation Script
==================
Evaluate trained models and generate publication-quality figures.

Produces:
  - Classification report (accuracy, precision, recall, F1, false alarm rate)
  - Confusion matrix plot
  - Per-class performance bar chart

Usage:
    python scripts/evaluate.py --model cnn_lstm
    python scripts/evaluate.py --model cnn_lstm --save_figures
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import tensorflow as tf


NUM_CLASSES = 5
CLASS_NAMES = [
    'Normal', 'SLG (A-G)', 'SLG (B-G)', 'SLG (C-G)', '3-Phase Short'
]

MODEL_PATHS = {
    'cnn_lstm': 'models/cnn_lstm_best.h5',
    'bilstm':   'models/bilstm_best.h5',
    'lstm':     'models/lstm_best.h5',
}


def compute_false_alarm_rate(y_true, y_pred):
    """
    False Alarm Rate = FP / (FP + TN) for the normal class (label 0).
    A false alarm is when the model predicts a fault when there is none.
    """
    normal_mask = (y_true == 0)
    if normal_mask.sum() == 0:
        return 0.0
    false_alarms = np.sum((y_true == 0) & (y_pred != 0))
    total_normal = normal_mask.sum()
    return false_alarms / total_normal


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Generate publication-quality confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, vmin=0, vmax=100,
                annot_kws={'size': 12})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (%)', fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_per_class_metrics(report_dict, class_names, save_path=None):
    """Bar chart of precision, recall, F1 per class."""
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        values = [report_dict[str(c)][metric] * 100 for c in range(len(class_names))]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_xlabel('Fault Class', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(bottom=max(0, min(ax.get_ylim()[0], 80)))
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def evaluate(args):
    """Main evaluation pipeline."""

    # Load test data
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
    print(f"Test set: {X_test.shape[0]} windows")

    # Load model
    model_path = MODEL_PATHS.get(args.model, args.model)
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Predict
    y_pred_proba = model.predict(X_test, batch_size=256, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    far = compute_false_alarm_rate(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"Model: {args.model.upper()}")
    print(f"{'='*50}")
    print(f"Accuracy:         {acc*100:.2f}%")
    print(f"False Alarm Rate: {far*100:.2f}%")
    print(f"\nClassification Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        digits=4, output_dict=True
    )
    print(classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        digits=4
    ))

    # Plots
    cm_path = f'figures/confusion_matrix_{args.model}.png' if args.save_figures else None
    bar_path = f'figures/per_class_{args.model}.png' if args.save_figures else None

    plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, save_path=cm_path)
    plot_per_class_metrics(report, CLASS_NAMES, save_path=bar_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate fault classification model')
    parser.add_argument('--model', type=str, default='cnn_lstm',
                        choices=['cnn_lstm', 'bilstm', 'lstm'],
                        help='Model to evaluate')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_figures', action='store_true',
                        help='Save figures to figures/ directory')
    args = parser.parse_args()

    evaluate(args)
