import numpy as np
import cv2
from keras.models import load_model
import os
import time

model = load_model("./models/model_v6_23.hdf5")
emotion_dict = {'Happy': 0, 'Sad': 5, 'Fear': 4, 'Disgust': 1, 'Surprise': 6, 'Neutral': 2, 'Angry': 3}

def process(directory, filename, logFileName):
    # a+ arg: 'a' means 'append', + means create if file doesn't exist
    log = open(logFileName, "a+")

    # loading image
    full_size_image = cv2.imread(directory + "\\" + filename)
    print(filename + " loaded. Recognizing...")

    start = time.time()
    emotion = recognize(full_size_image)
    end = time.time()
    diff = end - start

    res = str(diff) + "s /// " + filename + " /// " + emotion
    print(res)
    log.write(res + "\n")

    log.close()


def lfw():
    counter = 0
    directories = os.walk("C:\\Users\\weazy\\Desktop\\lfw")
    for directory in directories:
        if counter != 0:
            for files in os.walk(directory[0]):
                for file in files[2]:
                    process(directory[0], file, "lfw_result.txt")
        else:
            counter += 1


def celeba():
    path = "C:\\Users\\weazy\\Desktop\\celeba"
    directories = os.walk(path)

    for directory, subdirs, files in directories:
        for file in files:
            process(path, file, "celeba_result.txt")


def recognize(image):
    face_image = cv2.resize(image, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    predicted_class = np.argmax(model.predict(face_image))
    label_map = dict((v, k) for k, v in emotion_dict.items())
    predicted_label = label_map[predicted_class]

    return predicted_label

dataset = input("Input dataset (0 - LFW / 1 - CelebA): ")

if dataset == "0":
    lfw()
else:
    celeba()
