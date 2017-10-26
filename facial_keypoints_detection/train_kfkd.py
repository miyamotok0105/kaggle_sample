import os

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

print("X len " , len(X))
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))

for _x in X:
#     #<class 'numpy.ndarray'>
    if _x.dtype != "float32":
        print(_x.dtype)

for _y in y:
    #<class 'numpy.ndarray'>
    if _y.dtype != "float32":
        print(_y.dtype)

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(100, input_dim=9216))
model.add(Activation('relu'))
model.add(Dense(30))

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

# filepath="weights-{epoch:02d}.hdf5"
# checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
# model.fit(X, y, nb_epoch=200, validation_split=0.33, batch_size=10, callbacks=callbacks_list, verbose=0)

hist = model.fit(X, y, nb_epoch=10, validation_split=0.2)
y_test_proba = model.predict_proba(X)
print(y_test_proba)

model.save_weights("kfkd_model_10.h5")

from matplotlib import pyplot

X_test, _ = load(test=True)

print("X_test.shape ", X_test.shape)
print("X_test[0].shape ", X_test[0].shape)

y_test = model.predict(X_test)

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    axis = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    plot_sample(X_test[i], y_test[i], axis)

pyplot.show()


