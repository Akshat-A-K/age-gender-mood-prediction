"""Convert the training .h5 checkpoints into TensorFlow Lite files.

Download/place `Gender-age.h5` and `mood.h5` in the repo root, then run::

    python convert_models.py

Large artifacts (>50 MiB) are automatically chunked into
`models/chunks/<name>/name.partXX` so the repository avoids Git LFS quotas.
The plain `.tflite` files are rebuilt automatically at runtime when needed.
"""

from pathlib import Path

import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS = [
    (BASE_DIR / "Gender-age.h5", BASE_DIR / "models" / "age_gender.tflite"),
    (BASE_DIR / "mood.h5", BASE_DIR / "models" / "mood.tflite"),
]
CHUNK_SIZE = 50 * 1024 * 1024  # 50 MiB to stay below GitHub's file limit


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
    _chunk_large_artifact(target)


def _chunk_large_artifact(artifact: Path) -> None:
    if not artifact.exists():
        return

    size = artifact.stat().st_size
    if size <= CHUNK_SIZE:
        return

    chunk_dir = artifact.parent / "chunks" / artifact.stem
    chunk_dir.mkdir(parents=True, exist_ok=True)

    with artifact.open("rb") as src:
        idx = 0
        while True:
            data = src.read(CHUNK_SIZE)
            if not data:
                break
            part_path = chunk_dir / f"{artifact.stem}.part{idx:02d}"
            part_path.write_bytes(data)
            idx += 1

    artifact.unlink()
    print(
        f"Chunked {artifact.name} ({size / (1024 * 1024):.1f} MiB) into {idx} parts under "
        f"{chunk_dir.relative_to(BASE_DIR)}"
    )


def main() -> None:
    for checkpoint, artifact in CHECKPOINTS:
        convert_model(checkpoint, artifact)


if __name__ == "__main__":
    main()
