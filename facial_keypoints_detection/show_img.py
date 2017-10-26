import os
import cv2

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

#set "KERAS_BACKEND=tensorflow"


FTRAIN = './training.csv'
FTEST = './test.csv'


def load(test=False, cols=None):
    """testがTrueの場合はFTESTからデータを読み込み、Falseの場合はFTRAINから読み込みます。
    colsにリストが渡された場合にはそのカラムに関するデータのみ返します。
    """

    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname)) # pandasのdataframeを使用

    # スペースで句切られているピクセル値をnumpy arrayに変換
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # カラムに関連するデータのみを抽出
        df = df[list(cols) + ['Image']]

    print("df.count() ", df.count())  # カラム毎に値が存在する行数を出力
    df = df.dropna()  # データが欠けている行は捨てる

    X = np.vstack(df['Image'].values) / 255.  # 0から1の値に変換
    X = X.astype(np.float32)

    if not test:  # ラベルが存在するのはFTRAINのみ
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # -1から1の値に変換
        X, y = shuffle(X, y, random_state=42)  # データをシャッフル
        y = y.astype(np.float32)
        print("len(y) ", len(y))
    else:
        y = None

    return X, y

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    print("img.shape ", img.shape)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

X, y = load()

# for _x in X:
#     print(_x.dtype)
#     print(_x.shape)
#     break

img1 = X[0].reshape(96, 96)
img2 = X[1].reshape(96, 96)
img3 = X[2].reshape(96, 96)
img4 = X[3].reshape(96, 96)

while True:
    cv2.imshow('camera capture', img1)
    k = cv2.waitKey(2) # 1msec待つ
    if k == 27: # ESCキーで終了
        break
