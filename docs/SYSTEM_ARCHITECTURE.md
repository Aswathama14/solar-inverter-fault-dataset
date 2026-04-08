# System Architecture

## HIL Simulation Testbed

The dataset was generated using a Hardware-in-the-Loop (HIL) real-time simulation testbed at Texas A&M University-Kingsville, Department of Electrical Engineering and Computer Science.

### Hardware Components

| Component | Specification |
|-----------|---------------|
| **Real-Time Simulator** | OPAL-RT OP4510 |
| **Software** | MATLAB/Simulink R2024b + RT-LAB |
| **Data Acquisition** | Raspberry Pi 4 Model B (4 GB RAM) |
| **Communication** | MQTT protocol |
| **Sampling Rate** | 10 kHz |

### Inverter System Model

The grid-tied solar inverter model comprises five subsystems:

1. **PV Source**: Single-diode equivalent circuit model
2. **DC-DC Converter**: Boost converter with Incremental Conductance MPPT
3. **Three-Phase Inverter**: Two-level Voltage Source Inverter (VSI) with Space-Vector PWM (SVPWM)
4. **Output Filter**: LCL filter designed per IEEE 519 harmonic limits
5. **Grid Interface**: SOGI-PLL for synchronization + synchronous d-q frame PI current controller

### Key System Parameters

| Parameter | Value |
|-----------|-------|
| PV array rated power | Configured for study conditions |
| Grid voltage | 3-phase, 60 Hz |
| Switching frequency | 10 kHz |
| LCL filter | IEEE 519 compliant |
| MPPT algorithm | Incremental Conductance |
| PWM scheme | Space-Vector PWM (SVPWM) |
| PLL type | Second-Order Generalized Integrator (SOGI) |
| Current controller | Synchronous d-q frame PI with IMC tuning |

### Fault Injection

Faults were injected at the grid connection point using a programmable fault switch in the Simulink model. Each simulation scenario runs for 18 seconds. Multiple injection times (7 s, 9 s, 12 s, 16 s) provide different pre-fault and post-fault signal histories.

### Data Acquisition Pipeline

```
OPAL-RT OP4510 (Real-Time Simulation)
         │
         │ Analog/Digital I/O
         ▼
    MQTT Broker
         │
         │ MQTT Subscribe
         ▼
  Raspberry Pi 4 Model B
         │
         │ Python logging script
         ▼
  CSV files (time-stamped)
```

The Raspberry Pi subscribes to MQTT topics published by the OPAL-RT, logging 11 electrical signals at 10 kHz. This setup emulates an edge-device monitoring scenario used in practical renewable energy installations.

## Noise Augmentation

5% Gaussian noise (SNR ≈ 26 dB) is superimposed on all sensor signals to replicate realistic measurement uncertainty consistent with the OPAL-RT-to-Raspberry Pi data acquisition chain.

## References

For complete system details, see the associated conference paper:

> D. P. Patel, I. P. Pathak, D. Roach, and N. Yilmazer, "Robust Deep Learning Models for Fault Detection in Grid-Tied Solar Inverters: A Comparative Study of LSTM, Bi-LSTM, and CNN-LSTM Architectures," in Proc. IEEE Green Tech. Conf., Boulder, CO, USA, 2026. DOI: [10.1109/GreenTech68285.2026.11471570](https://doi.org/10.1109/GreenTech68285.2026.11471570)
