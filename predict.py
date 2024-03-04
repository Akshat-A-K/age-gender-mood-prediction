import cv2
import numpy as np
from keras.models import load_model, save_model


def getAge(distr):
    distr = distr*4
    if distr >= 0.65 and distr < 1.65:
        return "0-18"
    if distr >= 1.65 and distr < 2.65:
        return "19-30"
    if distr >= 2.65 and distr < 3.65:
        return "31-80"
    if distr >= 3.65 and distr < 4.65:
        return "80 +"
    return "Unknown"


def getGender(prob):
    if prob < 0.5:
        return "Male"
    else:
        return "Female"


def getAgeGender(image_path):
    image = cv2.imread(image_path,0)
    image = cv2.resize(image,dsize=(128,128))
    image = image.reshape((image.shape[0],image.shape[1],1))

    model = load_model('Gender-age.h5')

    image = image/255
    val = model.predict(np.array([image]))
    age = getAge(val[0])
    gender = getGender(val[1])
    return age, gender

print(getAgeGender('enhanced_resized_face.jpg'))