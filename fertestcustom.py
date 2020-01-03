# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os
import numpy as np
import cv2
import time

# loading the model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fer.h5")
print("Loaded model from disk")

# setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x = None
y = None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def process(directory, filename, logFileName):
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

    # a+ arg: 'a' means 'append', + means create if file doesn't exist

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
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = face.detectMultiScale(gray, 1.3, 10)

    emotion = ""
    # detecting faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        yhat = loaded_model.predict(cropped_img)
        emotion = labels[int(np.argmax(yhat))]

    return emotion


dataset = input("Input dataset (0 - LFW / 1 - CelebA): ")

if dataset == "0":
    lfw()
else:
    celeba()
