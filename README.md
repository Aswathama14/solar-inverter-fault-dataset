# Grid-Tied Solar Inverter Fault Detection Dataset

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FGreenTech68285.2026.11471570-blue)](https://doi.org/10.1109/GreenTech68285.2026.11471570)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset](https://img.shields.io/badge/Zenodo-Dataset-orange)](https://zenodo.org/record/XXXXXX)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

> **HIL-Validated Multi-Class Fault Detection Benchmark for Three-Phase Grid-Tied Solar Photovoltaic Inverters**

This repository contains the dataset, code, and pre-trained models from our IEEE GreenTech 2026 paper. The dataset provides **5-class fault detection** data generated via **Hardware-in-the-Loop (HIL) real-time simulation** on OPAL-RT OP4510, acquired through a Raspberry Pi 4 over MQTT — emulating a real-world edge-device monitoring scenario.

📄 **Paper**: [IEEE GreenTech 2026](https://doi.org/10.1109/GreenTech68285.2026.11471570)  
📦 **Dataset (Zenodo)**: [https://zenodo.org/records/19463598] (https://zenodo.org/records/19463598)  
🏫 **Affiliation**: Texas A&M University-Kingsville | NSF Award CNS-2219733

---

## Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Fault Classes (5-Class)](#fault-classes-5-class)
- [Feature Description](#feature-description)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Reproduce Results](#reproduce-results)
- [Pre-trained Models](#pre-trained-models)
- [Benchmarks](#benchmarks)
- [System Architecture](#system-architecture)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Fault detection in grid-tied solar inverters is critical for grid stability and equipment protection. Existing datasets are either simulation-only without hardware validation, or limited to binary (fault/no-fault) classification. This dataset addresses both gaps by providing:

- **5-class multi-fault classification** (not just binary detection)
- **Hardware-in-the-Loop validation** on OPAL-RT OP4510 real-time simulator
- **Edge-device data acquisition** via Raspberry Pi 4 + MQTT protocol
- **Realistic noise augmentation** (5% Gaussian, SNR ≈ 26 dB)
- **Multi-domain features** spanning time, frequency, reference-frame, and power domains
- **IEEE 519 compliant** signal characteristics

## Dataset Description

| Property | Value |
|----------|-------|
| **Total Samples** | ~9000005 sliding-window sequences |
| **Classes** | 5 (1 normal + 4 fault types) |
| **Features** | 11 electrical signals |
| **Sampling Rate** | 10 kHz |
| **Window Size** | 100 samples (10 ms) |
| **Step Size** | 20 samples (2 ms) |
| **Noise Level** | 5% Gaussian (SNR ≈ 26 dB) |
| **Simulation Platform** | OPAL-RT OP4510 |
| **Data Acquisition** | Raspberry Pi 4 Model B via MQTT |
| **Format** | CSV (raw) + NumPy arrays (processed) |

## Fault Classes (5-Class)

| Label | Fault Type | Phases Affected | Fault Resistance | Injection Time |
|:-----:|------------|-----------------|:----------------:|:--------------:|
| 0 | Normal Operation | None | N/A | 0–18 s (baseline) |
| 1 | Single Line-to-Ground (A–G) | Phase A | 1.0 Ω | 7 s |
| 2 | Single Line-to-Ground (B–G) | Phase B | 1.0 Ω | 9 s |
| 3 | Single Line-to-Ground (C–G) | Phase C | 1.0 Ω | 12 s |
| 4 | Three-Phase Short Circuit (A–B–C) | All Phases | 0.01 Ω | 12 s, 16 s |

### Class Distribution

| Label | Fault Class | Approx. Samples |
|:-----:|-------------|:---------------:|
| 0 | Normal Operation | ~1800001 |
| 1 | SLG Fault (A–G) | ~1800001 |
| 2 | SLG Fault (B–G) | ~1800001 |
| 3 | SLG Fault (C–G) | ~1800001 |
| 4 | Three-Phase Short Circuit | ~1800001 |
| **Total** | **All 5 Classes** | **9000005** |

## Feature Description

The 11 electrical features span four signal domains, providing a comprehensive view of the inverter operating state:

| # | Feature | Symbol | Unit | Domain | Fault Signature Captured |
|:-:|---------|:------:|:----:|:------:|--------------------------|
| 1 | Phase A Voltage | V_a | V | Time | Voltage collapse, sag/swell |
| 2 | Phase B Voltage | V_b | V | Time | Voltage unbalance detection |
| 3 | Phase C Voltage | V_c | V | Time | Three-phase asymmetry |
| 4 | Phase A Current | I_a | A | Time | Over-current, current imbalance |
| 5 | Phase B Current | I_b | A | Time | Current asymmetry |
| 6 | Phase C Current | I_c | A | Time | Phase loss detection |
| 7 | D-axis Current | I_d | A | Reference-frame | Active power disruption |
| 8 | Q-axis Current | I_q | A | Reference-frame | Reactive power injection anomaly |
| 9 | Active Power | P | W | Power | Power collapse at fault onset |
| 10 | Reactive Power | Q | VAR | Power | Energy transfer disruption |
| 11 | Total Harmonic Distortion | THD | % | Frequency | Harmonic fault signatures (IEEE 519) |

## Quick Start

### Option 1: Python (Recommended)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('data/inverter_fault_dataset.csv')

# Separate features and labels
X = df.drop(columns=['time', 'label']).values  # shape: (N, 11)
y = df['label'].values                          # shape: (N,)

# Z-score normalization (fit on training split only)
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Sliding window (W=100 samples = 10 ms at 10 kHz, S=20 step)
W, S = 100, 20
sequences, labels = [], []
for i in range(0, len(X_norm) - W, S):
    sequences.append(X_norm[i:i+W])
    labels.append(np.bincount(y[i:i+W].astype(int)).argmax())

X_seq = np.array(sequences)  # shape: (N_win, 100, 11)
y_seq = np.array(labels)     # shape: (N_win,)

print(f"Dataset: {X_seq.shape[0]} windows, {X_seq.shape[1]} timesteps, {X_seq.shape[2]} features")
print(f"Classes: {np.unique(y_seq)}")
```

### Option 2: MATLAB

```matlab
data = readtable('data/inverter_fault_dataset.csv');
X = table2array(data(:, 2:12));  % 11 features
y = data.label;                   % fault labels (0–4)

% Z-score normalization
X_norm = (X - mean(X)) ./ std(X);

% Sliding window (W=100, step=20)
W = 100; S = 20;
n_win = floor((size(X_norm,1) - W) / S);
X_seq = zeros(n_win, W, 11);
y_seq = zeros(n_win, 1);
for i = 1:n_win
    idx = (i-1)*S + 1;
    X_seq(i,:,:) = X_norm(idx:idx+W-1, :);
    [~, y_seq(i)] = max(histcounts(y(idx:idx+W-1), -0.5:1:4.5));
    y_seq(i) = y_seq(i) - 1;  % 0-indexed labels
end
```

### Option 3: Use Pre-processed NumPy Arrays

```python
X_train = np.load('data/X_train.npy')  # (N_train, 100, 11)
y_train = np.load('data/y_train.npy')  # (N_train,)
X_test = np.load('data/X_test.npy')    # (N_test, 100, 11)
y_test = np.load('data/y_test.npy')    # (N_test,)
```

## Repository Structure

```
solar-inverter-fault-dataset/
├── README.md                          # This file
├── CITATION.cff                       # Machine-readable citation (GitHub standard)
├── LICENSE                            # CC BY 4.0
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── inverter_fault_dataset.csv     # Raw time-series (N × 13 columns)
│   ├── X_train.npy                    # Pre-processed training features
│   ├── y_train.npy                    # Training labels
│   ├── X_test.npy                     # Pre-processed test features
│   ├── y_test.npy                     # Test labels
│   └── README_data.md                 # Data dictionary and format details
│
├── scripts/
│   ├── load_dataset.py                # Data loading + sliding window + normalization
│   ├── train_cnn_lstm.py              # Train CNN-LSTM model
│   ├── train_bilstm.py               # Train Bi-LSTM model
│   ├── train_lstm.py                  # Train LSTM baseline
│   ├── evaluate.py                    # Generate metrics, confusion matrices, plots
│   └── noise_robustness.py            # Noise robustness analysis (0–15%)
│
├── models/
│   ├── cnn_lstm_best.h5               # Best CNN-LSTM weights
│   ├── cnn_lstm_tflite/               # TFLite model (Raspberry Pi ready)
│   │   └── model.tflite
│   └── inference_raspi.py             # TFLite inference script for edge deployment
│
├── figures/
│   ├── confusion_matrix_cnn_lstm.png
│   ├── training_curves.png
│   ├── noise_robustness.png
│   └── system_architecture.png
│
├── docs/
│   ├── SYSTEM_ARCHITECTURE.md         # HIL simulation setup details
│   └── FAULT_SIGNATURES.md            # Physical explanation of each fault class
│
└── .github/
    └── ISSUE_TEMPLATE/
        └── bug_report.md
```

## Reproduce Results

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/solar-inverter-fault-dataset.git
cd solar-inverter-fault-dataset

# Install dependencies
pip install -r requirements.txt

# Train CNN-LSTM (reproduces Table IV in the paper)
python scripts/train_cnn_lstm.py --epochs 10 --batch_size 64

# Evaluate all models
python scripts/evaluate.py --model cnn_lstm

# Run noise robustness analysis
python scripts/noise_robustness.py --noise_levels 0 2.5 5 7.5 10 15
```

## Pre-trained Models

| Model | Val. Accuracy | Training Time | Parameters | TFLite Size |
|-------|:------------:|:-------------:|:----------:|:-----------:|
| **CNN-LSTM** | **99.01%** | 135 min | ~120K | 480 KB |
| Bi-LSTM | 98.52% | 52 min | ~95K | 380 KB |
| LSTM | 83.85% | 83 min | ~48K | 192 KB |

## Benchmarks

### Classification Performance (5-Class)

| Model | Accuracy | Precision | Recall | F1-Score | False Alarm Rate |
|-------|:--------:|:---------:|:------:|:--------:|:----------------:|
| **CNN-LSTM** | **99.01%** | **99.14%** | **99.01%** | **99.07%** | **0.27%** |
| Bi-LSTM | 98.52% | 98.60% | 98.52% | 98.55% | 0.42% |
| LSTM | 83.85% | 84.20% | 83.85% | 83.90% | 4.15% |

### Noise Robustness (CNN-LSTM)

| Noise Level | Accuracy |
|:-----------:|:--------:|
| 0% | 99.8% |
| 5% (training) | 99.01% |
| 7.5% | 98.0%+ |
| 10% | 96.8% |
| 15% | 93.2% |

## System Architecture

The HIL simulation testbed consists of:
- **PV Source**: Single-diode equivalent circuit model
- **DC-DC Converter**: Boost converter with Incremental Conductance MPPT
- **Inverter**: Three-phase two-level VSI with Space-Vector PWM (SVPWM)
- **Output Filter**: LCL filter (IEEE 519 compliant)
- **Grid Sync**: SOGI-PLL for grid synchronization
- **Controller**: Synchronous d-q frame PI current controller
- **Platform**: OPAL-RT OP4510 running MATLAB/Simulink R2024b + RT-LAB
- **Data Acquisition**: Raspberry Pi 4 Model B via MQTT protocol

For detailed system parameters and schematic, see [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md).

---

## Citation

If you use this dataset or code in your research, please cite both the paper and the dataset:

### Paper (BibTeX)

```bibtex

@INPROCEEDINGS{11471570,
  author={Patel, Darshan Pankajkumar and Prashant Pathak, Ishan and Roach, Dawson and Yilmazer, Nuri},
  booktitle={2026 IEEE Green Technologies Conference (GreenTech)}, 
  title={Robust Deep Learning Models for Fault Detection in Grid-tied Solar Inverters: A Comparative Study of LSTM, Bi-LSTM, and CNN-LSTM Architectures}, 
  year={2026},
  volume={},
  number={},
  pages={476-481},
  keywords={Circuits;Filtering;Filters;Voltage multipliers;Capacitors;MIMICs;Circuits and systems;Millimeter wave integrated circuits;Monolithic integrated circuits;Integrated circuits;CNN-LSTM hybrid model;Deep learning fault detection;Grid-tied solar inverter;Hardware-in-the-Loop (HIL) validation;Photovoltaic (PV) systems;Real-time condition monitoring;MQTT},
  doi={10.1109/GreenTech68285.2026.11471570}}

```

### Dataset (BibTeX)

```bibtex
@dataset{patel2026inverter_fault_dataset,
  title     = {Grid-Tied Solar Inverter Multi-Class Fault Detection Dataset: HIL-Validated Time-Series with OPAL-RT and Raspberry Pi (5-Class, 91,000 Samples)},
  author    = {Patel, Darshan Pankajkumar and Pathak, Ishan P. and Roach, D. and Yilmazer, Nuri},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {https://ieeexplore.ieee.org/document/11471570},
  url       = {https://zenodo.org/records/19463598},
  license   = {CC-BY-4.0}
}
```

### Plain Text (IEEE Format)

>D. P. Patel, I. Prashant Pathak, D. Roach and N. Yilmazer, "Robust Deep Learning Models for Fault Detection in Grid-tied Solar Inverters: A Comparative Study of LSTM, Bi-LSTM, and CNN-LSTM Architectures," 2026 IEEE Green Technologies Conference (GreenTech), Boulder, CO, USA, 2026, pp. 476-481, doi: 10.1109/GreenTech68285.2026.11471570. keywords: {Circuits;Filtering;Filters;Voltage multipliers;Capacitors;MIMICs;Circuits and systems;Millimeter wave integrated circuits;Monolithic integrated circuits;Integrated circuits;CNN-LSTM hybrid model;Deep learning fault detection;Grid-tied solar inverter;Hardware-in-the-Loop (HIL) validation;Photovoltaic (PV) systems;Real-time condition monitoring;MQTT},
>  [10.1109/GreenTech68285.2026.11471570](https://doi.org/10.1109/GreenTech68285.2026.11471570)

---

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the data for any purpose, provided you give appropriate credit.

## Acknowledgments

This work was supported by the **National Science Foundation** under Award No. **CNS-2219733**. Real-time HIL simulation was performed using the OPAL-RT OP4510 platform at the Department of Electrical Engineering and Computer Science, Texas A&M University-Kingsville.

---

<p align="center">
  <b>⭐ If you find this dataset useful, please star this repository and cite our paper! ⭐</b>
</p>
