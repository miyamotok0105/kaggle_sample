import datetime
import argparse
import time
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
#set "KERAS_BACKEND=tensorflow"
#python web_cam.py

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((5, 2), dtype=dtype)
	for i in range(0, 5):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords


def landmark_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (2, 2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (2, 2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(500))
	model.add(Activation('relu'))
	model.add(Dense(500))
	model.add(Activation('relu'))
	model.add(Dense(30))
	model.load_weights("kfkd_model_cnn_1000.h5")
	return model

def facial_landmark(frame, model):
	_frame = frame / 255.
	_frame = _frame.reshape(1, 96, 96, 1)
	y_test = model.predict(_frame)
	y_test_proba = model.predict_proba(_frame)
	print(len(y_test_proba[0]))
	print(sum(y_test_proba[0]))
	print(sum(y_test_proba[0])/len(y_test_proba[0]))

	for (x, y) in zip(y_test[0][0::2] * 48 + 48, y_test[0][1::2] * 48 + 48):
		# print("x ", x)
		# print("y ", y)
		cv2.circle(frame, (x, y), 1, (255, 0, 0), 1)

	return frame

def capture_camera(mirror=True, size=None):
    cap = cv2.VideoCapture(0) # 0はカメラのデバイス番号
    model = landmark_model()
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = (96, 96)
        frame = cv2.resize(frame, size)
        frame = facial_landmark(frame, model)
        if mirror is True:
            frame = frame[:,::-1]
        cv2.imshow('camera capture', frame)

        k = cv2.waitKey(2) # 1msec待つ
        if k == 27: # ESCキーで終了
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
	capture_camera()
