"""Convert the training .h5 checkpoints into TensorFlow Lite files.

Download/place `Gender-age.h5` and `mood.h5` in the repo root, then run::

    python convert_models.py

Large artifacts (>50 MiB) are automatically chunked into
`models/chunks/<name>/name.partXX` so the repository avoids Git LFS quotas.
The plain `.tflite` files are rebuilt automatically at runtime when needed.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS = [
    (
        BASE_DIR / "Gender-age.h5",
        BASE_DIR / "models" / "age_gender.tflite",
        False,  # keep float32 outputs identical to the original deployment
    ),
    (
        BASE_DIR / "mood.h5",
        BASE_DIR / "models" / "mood.tflite",
        True,
    ),
]
CHUNK_SIZE = 50 * 1024 * 1024  # 50 MiB to stay below GitHub's file limit
OUTPUT_ALIASES: Dict[str, List[str]] = {
    "age_gender.tflite": ["age", "gender"],
}


def convert_model(source: Path, target: Path, optimize: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing checkpoint: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(source, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    target.write_bytes(tflite_model)
    _write_output_signature(model, tflite_model, target)
    print(f"Saved {target.relative_to(BASE_DIR)} ({len(tflite_model) / 1024:.1f} KiB)")
    _chunk_large_artifact(target)


def _write_output_signature(model, tflite_blob: bytes, artifact: Path) -> None:
    aliases = OUTPUT_ALIASES.get(artifact.name)
    if not aliases or len(model.outputs) != len(aliases):
        return

    interpreter = tf.lite.Interpreter(model_content=tflite_blob)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]

    shape = [dim if dim is not None else 1 for dim in input_detail["shape"]]
    sample = np.random.rand(*shape).astype(input_detail["dtype"])  # deterministic enough for mapping

    keras_outputs = model.predict(sample, verbose=0)
    if not isinstance(keras_outputs, (list, tuple)):
        keras_outputs = [keras_outputs]

    interpreter.set_tensor(input_detail["index"], sample)
    interpreter.invoke()
    tflite_details = interpreter.get_output_details()
    tflite_outputs = [interpreter.get_tensor(detail["index"]) for detail in tflite_details]

    remaining = set(range(len(tflite_outputs)))
    mapping = {}
    for alias, keras_out in zip(aliases, keras_outputs):
        best_idx = None
        best_score = float("inf")
        for idx in remaining:
            score = float(np.max(np.abs(tflite_outputs[idx] - keras_out)))
            if score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            continue
        remaining.remove(best_idx)
        mapping[alias] = tflite_details[best_idx]["name"]

    if len(mapping) == len(aliases):
        signature_path = artifact.with_suffix(".signature.json")
        signature_path.write_text(json.dumps(mapping, indent=2))


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
    for checkpoint, artifact, optimize in CHECKPOINTS:
        convert_model(checkpoint, artifact, optimize)


if __name__ == "__main__":
    main()
