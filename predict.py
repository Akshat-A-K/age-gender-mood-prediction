import cv2
import numpy as np
from keras.models import load_model


AGE_GENDER_MODEL_PATH = 'Gender-age.h5'
MOOD_MODEL_PATH = 'mood.h5'


age_gender_model = load_model(AGE_GENDER_MODEL_PATH, compile=False)
mood_model = load_model(MOOD_MODEL_PATH, compile=False)

MOOD_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise',
}


def _get_age_label(distr):
    distr = distr * 4
    if distr >= 0.65 and distr < 1.65:
        return "0-18"
    if distr >= 1.65 and distr < 2.65:
        return "19-30"
    if distr >= 2.65 and distr < 3.65:
        return "31-80"
    if distr >= 3.65 and distr < 4.65:
        return "80 +"
    return "Unknown"


def _get_gender_label(prob):
    if prob < 0.5:
        return "Male"
    return "Female"


def _preprocess_image(image_path, size):
    image = cv2.imread(image_path, 0)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")
    image = cv2.resize(image, dsize=size)
    image = image.reshape((image.shape[0], image.shape[1], 1))
    return image / 255.0


def get_age_gender(image_path):
    image = _preprocess_image(image_path, (128, 128))
    age_pred, gender_pred = age_gender_model.predict(np.array([image]), verbose=0)
    age = _get_age_label(float(age_pred[0][0]))
    gender = _get_gender_label(float(gender_pred[0][0]))
    return age, gender


def get_mood(image_path):
    image = _preprocess_image(image_path, (48, 48))
    result = mood_model.predict(np.array([image]), verbose=0)[0]
    return MOOD_LABELS.get(int(np.argmax(result)), "Unknown")


def get_age_gender_mood(image_path):
    age, gender = get_age_gender(image_path)
    mood = get_mood(image_path)
    return age, gender, mood


if __name__ == "__main__":
    test_image = 'enhanced_resized_face.jpg'
    age, gender, mood = get_age_gender_mood(test_image)
    print(f"Age: {age}, Gender: {gender}, Mood: {mood}")