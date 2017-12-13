import datetime
import argparse
import time
import cv2
import dlib
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

def facial_landmark(frame, dst, x_face, y_face, w_face, h_face, model):
	dst = dst / 255.
	print(dst.shape)
	w_rate = dst.shape[0] / 96
	h_rate = dst.shape[1] / 96
	size = (96, 96)
	dst = cv2.resize(dst, size)
	dst = dst.reshape(1, 96, 96, 1)
	y_test = model.predict(dst)
	y_test_proba = model.predict_proba(dst)
	# print(len(y_test_proba[0]))
	# print(sum(y_test_proba[0]))
	# print(sum(y_test_proba[0])/len(y_test_proba[0]))

	for (x, y) in zip(y_test[0][0::2] * 48 + 48, y_test[0][1::2] * 48 + 48):
		print("x ", x)
		print("y ", y)
		print("w_rate ", w_rate)
		print("h_rate ", h_rate)
		cv2.circle(frame, (int((x*w_rate) + x_face), int((y*h_rate) + y_face)), 1, (255, 0, 0), 1)

	return frame

def capture_camera(mirror=True, size=None):
    cap = cv2.VideoCapture(0) # 0はカメラのデバイス番号
    detector = dlib.get_frontal_face_detector()
    model = landmark_model()
    while True:
        ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(frame, 1)
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
            x = d.left()
            y = d.top()
            w = d.right() - d.left()
            h = d.bottom() - d.top()
            dst = frame[y:y+h, x:x+w]
            frame = facial_landmark(frame, dst, x, y, w, h, model)
        
        cv2.imshow('camera capture', frame)

        k = cv2.waitKey(2) # 1msec待つ
        if k == 27: # ESCキーで終了
            break
    cap.release()
    cv2.destroyAllWindows()

# def facial_detection(frame, dets):
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     w = frame.shape[0]
#     h = frame.shape[1]
#     print("Number of faces detected: {}".format(len(dets)))
#     for i, d in enumerate(dets):
#         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
#         # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
#     return frame



if __name__ == '__main__':
	capture_camera()
