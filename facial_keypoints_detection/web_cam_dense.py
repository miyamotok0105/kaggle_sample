import datetime
import argparse
import time
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
#set "KERAS_BACKEND=tensorflow"
#python web_cam.py

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((5, 2), dtype=dtype)
	for i in range(0, 5):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def landmark_model():
	model = Sequential()
	model.add(Dense(100, input_dim=9216))
	model.add(Activation('relu'))
	model.add(Dense(30))
	model.load_weights("kfkd_model_200.h5")
	return model

def facial_landmark(frame, model):
	_frame = frame.flatten().reshape(1, 9216)
	_frame = _frame / 255.

	y_test = model.predict(_frame)
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
