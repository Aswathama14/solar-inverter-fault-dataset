"""
TFLite Inference Script for Raspberry Pi
==========================================
Real-time fault classification on edge devices using TensorFlow Lite.

Designed for Raspberry Pi 4 Model B with MQTT data acquisition from OPAL-RT.

Usage:
    python models/inference_raspi.py --model models/cnn_lstm_tflite/model.tflite
    python models/inference_raspi.py --model models/cnn_lstm_tflite/model.tflite --benchmark

Requirements (Raspberry Pi):
    pip install tflite-runtime numpy

Paper: Patel et al. (2026), IEEE GreenTech Conference
DOI: 10.1109/GreenTech68285.2026.11471570
"""

import argparse
import time
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


NUM_CLASSES = 5
WINDOW_SIZE = 100
NUM_FEATURES = 11

CLASS_NAMES = {
    0: 'Normal',
    1: 'SLG (A-G)',
    2: 'SLG (B-G)',
    3: 'SLG (C-G)',
    4: '3-Phase Short',
}


class FaultClassifier:
    """TFLite-based real-time fault classifier for edge deployment."""

    def __init__(self, model_path: str):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]['shape']
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {input_shape}, dtype: {self.input_details[0]['dtype']}")

    def predict(self, window: np.ndarray) -> tuple:
        """
        Classify a single window of sensor data.

        Parameters
        ----------
        window : array of shape (100, 11), Z-score normalized

        Returns
        -------
        (predicted_class, confidence, class_name)
        """
        input_data = np.expand_dims(window, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        predicted = np.argmax(output)
        confidence = output[predicted]
        return predicted, confidence, CLASS_NAMES[predicted]

    def benchmark(self, n_iterations: int = 1000) -> dict:
        """Benchmark inference latency."""
        dummy = np.random.randn(WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
        latencies = []

        # Warmup
        for _ in range(10):
            self.predict(dummy)

        # Benchmark
        for _ in range(n_iterations):
            start = time.perf_counter()
            self.predict(dummy)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        latencies = np.array(latencies)
        results = {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'p50_ms': np.median(latencies),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'throughput_hz': 1000.0 / np.mean(latencies),
        }

        print(f"\nBenchmark ({n_iterations} iterations):")
        print(f"  Mean latency:  {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
        print(f"  P50 latency:   {results['p50_ms']:.2f} ms")
        print(f"  P95 latency:   {results['p95_ms']:.2f} ms")
        print(f"  Throughput:    {results['throughput_hz']:.0f} inferences/sec")
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFLite fault classifier')
    parser.add_argument('--model', type=str,
                        default='models/cnn_lstm_tflite/model.tflite')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run latency benchmark')
    args = parser.parse_args()

    classifier = FaultClassifier(args.model)

    if args.benchmark:
        classifier.benchmark()
    else:
        # Demo: classify random window
        demo_window = np.random.randn(WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
        label, conf, name = classifier.predict(demo_window)
        print(f"\nPrediction: {name} (label={label}, confidence={conf:.4f})")
