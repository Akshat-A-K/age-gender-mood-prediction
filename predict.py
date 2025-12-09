import cv2
import numpy as np
from pathlib import Path

try:  # Use TensorFlow's bundled Lite interpreter when available.
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
except ImportError:  # pragma: no cover - fallback when tensorflow isn't installed.
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except ImportError as exc:  # pragma: no cover - surface clearer guidance.
        raise ImportError(
            "No TensorFlow Lite interpreter found. Install tensorflow-cpu or tflite-runtime."
        ) from exc


BASE_DIR = Path(__file__).resolve().parent
AGE_GENDER_MODEL_PATH = BASE_DIR / 'models' / 'age_gender.tflite'
MOOD_MODEL_PATH = BASE_DIR / 'models' / 'mood.tflite'


class TFLiteModel:
    def __init__(self, model_path: Path):
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, image_tensor: np.ndarray):
        data = image_tensor.astype(self.input_details[0]['dtype'], copy=False)
        if data.ndim == len(self.input_details[0]['shape']) - 1:
            data = np.expand_dims(data, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], data)
        self.interpreter.invoke()
        return [self.interpreter.get_tensor(detail['index']) for detail in self.output_details]


age_gender_model = TFLiteModel(AGE_GENDER_MODEL_PATH)
mood_model = TFLiteModel(MOOD_MODEL_PATH)
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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


def _extract_face(gray_image):
    faces = FACE_CASCADE.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return gray_image
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    return gray_image[y:y + h, x:x + w]


def _preprocess_frame(image, size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = _extract_face(gray)
    face = cv2.equalizeHist(face)
    face = cv2.resize(face, dsize=size)
    face = face.reshape((face.shape[0], face.shape[1], 1))
    return face.astype("float32") / 255.0


def _preprocess_image(image_path, size):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")
    return _preprocess_frame(image, size)


def _preprocess_bytes(payload, size):
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image data.")
    return _preprocess_frame(image, size)


def _predict_age_gender(image_tensor):
    age_pred, gender_pred = age_gender_model.predict(image_tensor)
    age = _get_age_label(float(age_pred[0][0]))
    gender = _get_gender_label(float(gender_pred[0][0]))
    return age, gender


def _predict_mood(image_tensor):
    result = mood_model.predict(image_tensor)[0][0]
    return MOOD_LABELS.get(int(np.argmax(result)), "Unknown")


def get_age_gender(image_path):
    image = _preprocess_image(image_path, (128, 128))
    return _predict_age_gender(image)


def get_mood(image_path):
    image = _preprocess_image(image_path, (48, 48))
    return _predict_mood(image)


def get_age_gender_mood(image_path):
    age, gender = get_age_gender(image_path)
    mood = get_mood(image_path)
    return age, gender, mood


def get_age_gender_mood_from_bytes(image_bytes: bytes):
    age_image = _preprocess_bytes(image_bytes, (128, 128))
    mood_image = _preprocess_bytes(image_bytes, (48, 48))
    age, gender = _predict_age_gender(age_image)
    mood = _predict_mood(mood_image)
    return age, gender, mood


if __name__ == "__main__":
    test_image = 'enhanced_resized_face.jpg'
    age, gender, mood = get_age_gender_mood(test_image)
    print(f"Age: {age}, Gender: {gender}, Mood: {mood}")