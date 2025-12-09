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
- `Gender-age.h5` / `mood.h5` - pre-trained weights required for inference.

## Setup

1. Install Python 3.10 or newer.
2. Install the dependencies:
   ```bash
   python -m pip install tensorflow keras opencv-python numpy pandas matplotlib seaborn scikit-learn
   ```
3. Ensure a webcam is available for `capture.py` and that the `.h5` model files live in the project root.

## Usage

1. **Capture a face crop**
   ```bash
   python capture.py
   ```
   Press `c` in the preview window to save `captured_image.jpg`, `cropped_face.jpg`, `resized_face.jpg`, and `enhanced_resized_face.jpg`.

2. **Predict age, gender, mood**
   ```bash
   python predict.py
   ```
   The script loads `enhanced_resized_face.jpg` (and internally prepares the 48x48 tensor for the mood model) and prints all three predictions.

## Datasets

- Age and gender: https://www.kaggle.com/datasets/jangedoo/utkface-new
- Mood / emotion: https://www.kaggle.com/datasets/msambare/fer2013
