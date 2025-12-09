"""Utility to convert the training .h5 checkpoints into TensorFlow Lite files.

Run once locally:
    python convert_models.py
The script drops the converted artifacts into the ./models directory.
"""

from pathlib import Path

import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS = [
    (BASE_DIR / "Gender-age.h5", BASE_DIR / "models" / "age_gender.tflite"),
    (BASE_DIR / "mood.h5", BASE_DIR / "models" / "mood.tflite"),
]


def convert_model(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing checkpoint: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(source, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    target.write_bytes(tflite_model)
    print(f"Saved {target.relative_to(BASE_DIR)} ({len(tflite_model) / 1024:.1f} KiB)")


def main() -> None:
    for checkpoint, artifact in CHECKPOINTS:
        convert_model(checkpoint, artifact)


if __name__ == "__main__":
    main()
