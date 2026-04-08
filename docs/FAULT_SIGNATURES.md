# Fault Signature Analysis

Physical description of each fault class in the dataset, explaining how each fault manifests in the 11 electrical features.

## Label 0: Normal Operation

Under normal conditions, the inverter produces balanced three-phase sinusoidal output:
- Phase voltages (Va, Vb, Vc) are equal magnitude, 120° apart
- Phase currents (Ia, Ib, Ic) are balanced
- D-axis current (Id) is near-constant DC (active power component)
- Q-axis current (Iq) is near-zero (unity power factor)
- Active power (P) is approximately constant at the Maximum Power Point
- Reactive power (Q) is approximately zero
- THD is below 5% per IEEE 519

## Labels 1–3: Single Line-to-Ground (SLG) Faults

SLG faults are the most common fault type in T&D systems (~70–80% of all faults). When a single phase contacts ground through 1.0 Ω fault resistance:

**Key signatures:**
- **Faulted phase current spike**: Rapid rise in the affected phase current
- **Voltage collapse**: Faulted phase voltage drops
- **d-q axis oscillations**: 120 Hz oscillations (double fundamental frequency) appear in Id and Iq due to negative-sequence and zero-sequence current components
- **THD spike**: Sudden increase from superposition of odd-order harmonics
- **Active power drop**: 15–30% reduction due to voltage collapse

**Inter-class distinction:**
- Labels 1 (A-G), 2 (B-G), and 3 (C-G) are structurally identical up to cyclic phase permutation
- A-G faults affect features Va, Ia; B-G affects Vb, Ib; C-G affects Vc, Ic
- These are the most challenging classes to distinguish from each other

## Label 4: Three-Phase Short Circuit (A-B-C)

The most severe and distinctive fault type, modeled with 0.01 Ω fault resistance:

**Key signatures:**
- **Symmetric current surge**: All three phase currents increase 5–8× nominal simultaneously
- **Complete voltage collapse**: All three phase voltages drop to near zero
- **Active power collapse**: P drops to near zero within half a cycle
- **Extreme THD increase**: Very large harmonic content
- **Symmetric d-q response**: Unlike SLG faults, the d-q response maintains relative symmetry

This fault is the most reliably detected across all three model architectures due to its extreme and symmetric signal characteristics.

## Fault Injection Timeline

```
Time (s):  0 ──── 7 ──── 9 ──── 12 ──── 16 ──── 18
           │      │      │       │       │       │
           │  AG fault  BG fault │    3-Phase   │
           │  (Label 1) (Label 2)│   (Label 4)  │
           │              CG fault              │
           │             (Label 3)               │
           │                     3-Phase         │
           │                    (Label 4)        │
           ├──────────────────────────────────────┤
           Normal operation baseline (Label 0)
```

## Feature Domain Summary

| Domain | Features | Primary Information |
|--------|----------|---------------------|
| **Time** | Va, Vb, Vc, Ia, Ib, Ic | Direct waveform distortion, phase imbalance |
| **Reference-frame** | Id, Iq | Active/reactive power decoupling, 120 Hz oscillations |
| **Power** | P, Q | Energy transfer disruption magnitude |
| **Frequency** | THD | Harmonic contamination level (IEEE 519 metric) |

The multi-domain feature set ensures that faults are captured through complementary signal representations, improving classification robustness even under noisy conditions.

## References

> D. P. Patel, I. P. Pathak, D. Roach, and N. Yilmazer, "Robust Deep Learning Models for Fault Detection in Grid-Tied Solar Inverters," IEEE GreenTech 2026. DOI: [10.1109/GreenTech68285.2026.11471570](https://doi.org/10.1109/GreenTech68285.2026.11471570)
