# Age, Mood and Gender Prediction using Deep Learning

## Overview
This project now predicts three attributes from a single face crop:

- **Age group** (0-18, 19-30, 31-80, 80+) using a CNN trained on the UTKFace dataset
- **Gender** (male or female) from the same multitask model
- **Mood / facial expression** across seven FER emotion classes (`Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`)

The typical workflow is:
1. Capture a frame from the webcam, detect a face, and create grayscale crops with histogram equalization (`capture.py`).
2. Feed the 128x128 enhanced crop to the age/gender model and a 48x48 resize to the mood model (`predict.py`).
3. Read the predicted age bracket, gender, and mood in the console output.

## Repository Contents

- `age-gender.ipynb` - training notebook for the age and gender multitask CNN (UTKFace).
- `mood.ipynb` - training notebook for the FER-based mood classifier.
- `capture.py` - webcam capture, Haar face detection, grayscale conversion, histogram equalization, and artifact saving.
- `predict.py` - loads `Gender-age.h5` plus `mood.h5`, preprocesses an input face image, and prints all three attributes.
- `app.py` + `templates/` + `static/` - Flask application that exposes a browser UI for uploading face crops, capturing snapshots via webcam, and visualizing predictions.
- `Gender-age.h5` / `mood.h5` - original Keras checkpoints (download separately and place in the project root when you need to regenerate `.tflite` files).
- `models/chunks/age_gender/age_gender.partXX` - chunked TensorFlow Lite weights for age/gender inference. These ship with the repo and are reassembled automatically at runtime.

## Setup

1. Install Python 3.10 or newer.
2. Install the dependencies (or simply `pip install -r requirements.txt`):
   ```bash
   python -m pip install tensorflow keras opencv-python numpy pandas matplotlib seaborn scikit-learn flask
   ```
3. Ensure a webcam is available for `capture.py` and that the `.h5` model files live in the project root.

## Usage

1. **Capture a face crop**
   ```bash
   python capture.py
   ```
   Press `c` in the preview window to save `captured_image.jpg`, `cropped_face.jpg`, `resized_face.jpg`, and `enhanced_resized_face.jpg`.

2. **Predict age, gender, mood (CLI)**
   ```bash
   python predict.py
   ```
   The script loads `enhanced_resized_face.jpg` (and internally prepares the 48x48 tensor for the mood model) and prints all three predictions.

3. **Use the Flask web UI**
   ```bash
   python app.py
   ```
   - Browse to the printed `http://127.0.0.1:5000/` address. The landing page mirrors the webcam feed, offers a drag-and-drop uploader, and renders predictions in real time.
   - Use the **Enable Camera** and **Capture Snapshot** buttons inside the “Face Insight Studio” stage. Bounding boxes (when the browser supports the FaceDetector API) help you center your face before grabbing a still.
   - The right-hand column shows the upload panel, latest prediction preview, and a rolling session log so you can compare previous captures.
   - **Privacy controls**:
     - Toggle **“Do not persist captures”** to keep images in-memory only. Predictions still work, but the snapshots are never written to disk and won’t show up in the saved session log.
     - Use the **Clear Session** button to delete the current history and wipe the `uploads/` directory. This is useful for demos or if you enabled persistence temporarily and want to remove artifacts afterward.
   - Theme toggle (Noir/Aurora) lets you switch between dark and light UI palettes without restarting the server.

## Model artifacts

- The repository ships with the TensorFlow Lite artifacts only: the smaller mood model is checked in directly, while the age/gender model is stored as chunk files under `models/chunks/age_gender/`. The `.tflite` file is reconstructed on first run, so no Git LFS bandwidth is needed during deployment.
- At runtime we now load **TensorFlow Lite** versions (`models/age_gender.tflite`, `models/mood.tflite`) through TensorFlow's built-in Lite interpreter (`tensorflow-cpu`). This keeps inference lightweight while remaining compatible across hosts.
- If you retrain the models, regenerate the TFLite artifacts once using the helper script:
   ```bash
   python convert_models.py
   ```
   The script writes `.tflite` files into `./models/` so the web app can consume them. The age/gender export deliberately skips post-training quantization to keep its float32 output distribution identical to the previously deployed model. If you need a smaller file, provide a representative dataset and enable quantization manually.
   Artifacts larger than 50 MiB are chunked automatically into `models/chunks/<name>/` to keep Git-friendly file sizes; `predict.py` rebuilds the `.tflite` file from those pieces when needed.

## Deployment notes

- Free hosts such as Render require HTTPS for webcam access. Set the Python version to **3.10** (the repo includes `runtime.txt`) so prebuilt `tensorflow-cpu` wheels are available.
- Both Linux hosts (Render) and Windows development environments install `tensorflow-cpu==2.20.0`, which exposes the same TF Lite interpreter used for inference and model conversion. No Git LFS traffic is required because the age/gender weights are shipped as regular Git blobs split into 50 MiB parts.
- `Procfile` launches Gunicorn with `--workers=2 --timeout=120`, which mirrors setting `WEB_CONCURRENCY=2` on Render so a second worker can serve requests while another performs long-running inference. Adjust those flags if your host provides more/less RAM.

## Datasets

- Age and gender: https://www.kaggle.com/datasets/jangedoo/utkface-new
- Mood / emotion: https://www.kaggle.com/datasets/msambare/fer2013
