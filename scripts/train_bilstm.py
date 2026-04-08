"""
Bi-LSTM Training Script
========================
Trains the Bidirectional LSTM model for 5-class inverter fault classification.

Paper: Patel et al. (2026), IEEE GreenTech Conference
DOI: 10.1109/GreenTech68285.2026.11471570

Usage:
    python scripts/train_bilstm.py --epochs 10
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


NUM_CLASSES = 5
WINDOW_SIZE = 100
NUM_FEATURES = 11


def build_bilstm(input_shape=(WINDOW_SIZE, NUM_FEATURES), num_classes=NUM_CLASSES):
    """Bidirectional LSTM for time-series fault classification."""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.3),
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
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(args.data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(args.data_dir, 'y_val.npy'))

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)

    model = build_bilstm()
    model.summary()

    os.makedirs(args.model_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(os.path.join(args.model_dir, 'bilstm_best.h5'),
                        monitor='val_accuracy', mode='max', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, verbose=1
    )

    model.save(os.path.join(args.model_dir, 'bilstm_final.h5'))
    print(f"\n✓ Bi-LSTM training complete. Model saved to {args.model_dir}/")
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Bi-LSTM fault classifier')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    train(args)
