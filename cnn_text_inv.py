import time
import os

import numpy as np
import thulac
import pandas as pd
import gensim
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

import transcribe_common as Kc


def main():
    path_net = os.path.join('/', 'public', 'home', 'liqi', 'data', 'analysis', 'transcribe_CNN')
    # path_net = os.path.join('Z:', 'datas', 'analysis')

    datas = pd.read_csv(os.path.join(path_net, "save_CNN_text_2_datas.csv"))

    text_xys = Kc.DataFrameToXY(datas, ["HistoryOfPastIllness"], "MajorDiagnosisCoding",
                                manualFilePath=os.path.join(path_net, r"CodingToClass_1to16 - new 多分类.csv"))

    text_xys = text_xys[text_xys["MajorDiagnosisCoding"] != "-1"]

    # 字符串向量化
    text_x_all = ""
    text_x_thus = []
    text_x_d2v_train = []
    thu1 = thulac.thulac(seg_only=True)  # 默认模式
    for i in range(len(text_xys["HistoryOfPastIllness"])):
        try:
            index_text_y = text_xys["HistoryOfPastIllness"][i]
        except KeyError:
            index_text_y = ""

        # 分词
        temp_thu1_rt = thu1.cut(index_text_y, text=True)
        document = gensim.models.doc2vec.TaggedDocument(temp_thu1_rt, tags=[i])
        text_x_d2v_train.append(document)

    model = gensim.models.Doc2Vec(text_x_d2v_train, size=50, window=8, min_count=5, workers=4)
    model.save(os.path.join(path_net, r'model_text_2.model'))
    text_x_vds = list(map(lambda x: np.array(x), model.docvecs.vectors_docs.tolist()))
    text_xys.insert(1, 'HistoryOfPastIllness_vec', text_x_vds)
    text_xys.drop(['HistoryOfPastIllness'], axis=1, inplace=True)

    text_xys_np = text_xys.values

    x_train, y_train, x_test, y_test = Kc.SplitGroup(text_xys_np)
    y_train = y_train.astype(np.float64) - 1  # 没有得到的消除，只有1和2两类
    y_test = y_test.astype(np.float64) - 1  # 没有得到的消除，只有1和2两类
    x_train = np.array(list(map(lambda x: x[0].tolist(), x_train.tolist())))
    x_test = np.array(list(map(lambda x: x[0].tolist(), x_test.tolist())))

    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    numclasses = y_train.max() + 1
    onesharp = x_train.shape[2]

    # 创建神经网络
    print("搭建神经网络")
    model = tf.keras.Sequential(
        [layers.LSTM(16, return_sequences=True, input_shape=(1, onesharp)),
         layers.Conv1D(32, 5, padding='same', activation=tf.nn.relu),
         layers.Conv1D(64, 5, padding='same', activation=tf.nn.relu),
         # layers.MaxPooling1D(5),
         layers.Dropout(0.25),
         layers.LSTM(64, return_sequences=True),
         layers.Conv1D(128, 5, padding='same', activation=tf.nn.relu),
         layers.Conv1D(64, 5, padding='same', activation=tf.nn.relu),
         # layers.MaxPooling1D(5),
         layers.Dropout(0.25),
         layers.LSTM(64, return_sequences=True),
         layers.Conv1D(32, 5, padding='same', activation=tf.nn.relu),
         layers.GlobalAveragePooling1D(),
         layers.Dropout(0.5),
         layers.Dense(numclasses, activation=tf.nn.softmax)])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    print("训练神经网络")
    metrics = Kc.Metrics(validation_data=(x_test, y_test))
    history = model.fit(x_train, y_train, batch_size=400, validation_data=(x_test, y_test), epochs=500,
                        callbacks=[metrics, TensorBoard(log_dir=os.path.join('logs', '{}').format("模型名-{}".format(int(time.time()))))])

    nowtime = str(int(time.time()))
    try:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.plot(metrics.val_f1s)
        plt.plot(metrics.val_recalls)
        plt.plot(metrics.val_precisions)
        plt.legend(['training', 'validation', 'val_f1', 'val_recall', 'val_precision'], loc='upper left')

        plt.savefig(os.path.join(path_net, 'result', 'CNN_text_2', nowtime + '_cnn_result' + '.png'))
        plt.show()
    except:
        print("无法画图")
    finally:
        with open(os.path.join(path_net, 'result', 'CNN_text_2', nowtime + '_history.txt'), 'w+') as f:
            f.write(str(history.history))
    f = open('cnn_text.py', 'r', encoding='utf-8')
    fff = f.read()
    f.close()
    nf = open(os.path.join(path_net, 'result', 'CNN_text_2', nowtime + '_code' + '.py'), 'w+')
    nf.write(fff)
    nf.close()
    print('结束')
    return 0


if __name__ == '__main__':
    main()
