"""
CNN-LSTM Training Script
=========================
Trains the 1D CNN-LSTM model for 5-class inverter fault classification.
Reproduces Table IV results from:

    Patel et al. (2026), "Robust Deep Learning Models for Fault Detection
    in Grid-Tied Solar Inverters," IEEE GreenTech Conference.
    DOI: 10.1109/GreenTech68285.2026.11471570

Usage:
    python scripts/train_cnn_lstm.py
    python scripts/train_cnn_lstm.py --epochs 20 --batch_size 32
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical

# Allow relative imports when run from repo root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Hyperparameters ─────────────────────────────────────────────────────────
NUM_CLASSES = 5
WINDOW_SIZE = 100     # timesteps per sequence
NUM_FEATURES = 11     # electrical features


def build_cnn_lstm(input_shape=(WINDOW_SIZE, NUM_FEATURES),
                   num_classes=NUM_CLASSES) -> tf.keras.Model:
    """
    1D CNN-LSTM architecture for time-series fault classification.

    Architecture:
        Conv1D(64, k=5) → BN → MaxPool → Dropout
        Conv1D(128, k=3) → BN → MaxPool → Dropout
        LSTM(64) → Dropout
        Dense(32) → Dense(num_classes, softmax)

    Parameters
    ----------
    input_shape : tuple, (timesteps, features)
    num_classes : int, number of fault classes

    Returns
    -------
    tf.keras.Model
    """
    model = Sequential([
        # Block 1: Temporal feature extraction
        Conv1D(64, kernel_size=5, activation='relu', padding='same',
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Block 2: Higher-level feature extraction
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Block 3: Temporal sequence modeling
        LSTM(64, return_sequences=False),
        Dropout(0.3),

        # Classification head
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train(args):
    """Train CNN-LSTM model."""

    # ─── Load data ────────────────────────────────────────────────────────
    print("Loading pre-processed data...")
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(args.data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(args.data_dir, 'y_val.npy'))

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Classes: {np.unique(y_train)}")

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)

    # ─── Build model ──────────────────────────────────────────────────────
    model = build_cnn_lstm()
    model.summary()

    # ─── Callbacks ────────────────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            os.path.join(args.model_dir, 'cnn_lstm_best.h5'),
            monitor='val_accuracy', mode='max',
            save_best_only=True, verbose=1
        ),
        EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    # ─── Train ────────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # ─── Save final model ─────────────────────────────────────────────────
    model.save(os.path.join(args.model_dir, 'cnn_lstm_final.h5'))
    print(f"\nModel saved to {args.model_dir}/")

    # ─── Export to TFLite for Raspberry Pi deployment ─────────────────────
    if args.export_tflite:
        tflite_dir = os.path.join(args.model_dir, 'cnn_lstm_tflite')
        os.makedirs(tflite_dir, exist_ok=True)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = os.path.join(tflite_dir, 'model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {tflite_path} ({len(tflite_model)/1024:.0f} KB)")

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN-LSTM fault classifier')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing .npy files')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--export_tflite', action='store_true',
                        help='Export model to TFLite format')
    args = parser.parse_args()

    model, history = train(args)
    print("\n✓ Training complete. Run 'python scripts/evaluate.py' to see full metrics.")
