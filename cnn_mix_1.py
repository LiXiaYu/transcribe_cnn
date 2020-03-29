import time
import os
import re
import typing

import numpy as np
import thulac
import pandas as pd
import gensim
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

import transcribe_common as Kc


def main():
    path_net = os.path.join('/', 'public', 'home', 'liqi', 'data', 'analysis', 'transcribe_CNN')
    # path_net = os.path.join('Z:', 'datas', 'analysis')

    # xys_c_end = PreprocessData(path_net)

    # xys_c_end.to_csv(os.path.join(path_net, r'xys_c_end.csv'))
    xys_c_end = pd.read_csv(os.path.join(path_net, r'xys_c_end.csv'))

    xys_c_end['HistoryOfPastIllness_vec'] = xys_c_end['HistoryOfPastIllness_vec'].apply(lambda x: eval(re.sub(r'(?<=\d)\s+', ',', x, flags=re.S)))

    ###########
    # 转为np.array
    xys_dd = xys_c_end.values
    x_train, y_train, x_test, y_test = Kc.SplitGroup(xys_dd)
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)

    # 分离数据
    x_train_text = np.array(list(map(lambda x: x[13], x_train)))
    x_train_numerics = np.array(list(map(lambda x: x[0:12], x_train)), dtype=np.float64)
    y_train_all = y_train
    x_test_text = np.array(list(map(lambda x: x[13], x_test)))
    x_test_numerics = np.array(list(map(lambda x: x[0:12], x_test)), dtype=np.float64)
    y_test_all = y_test

    x_train_text = x_train_text.reshape(x_train_text.shape[0], 1, x_train_text.shape[1])
    x_train_numerics = x_train_numerics.reshape(x_train_numerics.shape[0], 1, x_train_numerics.shape[1])
    x_test_text = x_test_text.reshape(x_test_text.shape[0], 1, x_test_text.shape[1])
    x_test_numerics = x_test_numerics.reshape(x_test_numerics.shape[0], 1, x_test_numerics.shape[1])

    numclasses = y_train.max() + 1

    # 创建神经网络
    print("搭建神经网络")

    model = DefineModel_textAndNumerics([x_train_text.shape[2], x_train_numerics.shape[2]], numclasses)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    print("训练神经网络")
    metrics = Kc.Mertics_f1s_recalls_precisions(validation_data=([x_test_text, x_test_numerics], y_test))
    history = model.fit([x_train_text, x_train_numerics], y_train, batch_size=200, validation_data=([x_test_text, x_test_numerics], y_test), epochs=700,
                        callbacks=[metrics])

    nowtime = str(int(time.time()))
    try:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.plot(metrics.val_f1s)
        plt.plot(metrics.val_recalls)
        plt.plot(metrics.val_precisions)
        plt.legend(['training', 'validation', 'val_f1', 'val_recall', 'val_precision'], loc='upper left')

        plt.savefig(os.path.join(path_net, 'result', 'cnn_mix_1', nowtime + '_cnn_result' + '.png'))
        plt.show()
    except:
        print("无法画图")
    finally:
        with open(os.path.join(path_net, 'result', 'cnn_mix_1', nowtime + '_history.txt'), 'w+') as f:
            f.write(str(history.history))

        model.save(os.path.join(path_net, 'model', 'cnn_mix_1', nowtime + r'_cnn_mix_1.model'))

    f = open('cnn_text.py', 'r', encoding='utf-8')
    fff = f.read()
    f.close()
    nf = open(os.path.join(path_net, 'result', 'cnn_mix_1', nowtime + '_code' + '.py'), 'w+')
    nf.write(fff)
    nf.close()
    print('结束')

    return 0


def PreprocessData(path_net):
    datas = pd.read_csv(os.path.join(path_net, "save_CNN_text_2_datas.csv"))
    numeric_xys = Kc.DataFrameToXY(datas, ["Sex", "Years", "Temperature", "Pulse", "Breath", "Blood_Pressure_Systolic",
                                           "Blood_Pressure_Diastolic", "Width", "Height", "Corneal_Endothelium_Right",
                                           "Corneal_Endothelium_Left", "Right_Intraocular_Pressure",
                                           "Left_Intraocular_Pressure"],
                                   "MajorDiagnosisCoding",
                                   manualFilePath=os.path.join(path_net, r"CodingToClass_1to16 - new 多分类.csv"))
    text_xys = Kc.DataFrameToXY(datas, ["HistoryOfPastIllness"], "MajorDiagnosisCoding",
                                manualFilePath=os.path.join(path_net, r"CodingToClass_1to16 - new 多分类.csv"))
    # # 排除不要的标记数据
    numeric_xys = numeric_xys[numeric_xys["MajorDiagnosisCoding"] != "-1"]
    text_xys = text_xys[text_xys["MajorDiagnosisCoding"] != "-1"]
    # 字符串向量化
    model, text_x_vds = Kc.TextToVector_doc2vec(text_xys["HistoryOfPastIllness"].values.tolist())
    model.save(os.path.join(path_net, r'model_text_2.model'))
    text_xys.insert(1, 'HistoryOfPastIllness_vec', text_x_vds)
    text_xys.drop(['HistoryOfPastIllness'], axis=1, inplace=True)
    text_xys_np = text_xys.values
    # 数值型插补
    numeric_xys = Kc.Mice_numeric(numeric_xys)
    # # 合并数据
    text_xys["MajorDiagnosisCoding"] = text_xys["MajorDiagnosisCoding"].astype("float64")
    text_xys = text_xys.reset_index(drop=True)
    all_xys = numeric_xys.copy()
    all_xys['HistoryOfPastIllness_vec'] = text_xys['HistoryOfPastIllness_vec']
    # 数据校验
    # # 取出低压
    xys_c_0 = all_xys.copy()
    # xys_c_0.loc[:, 'MajorDiagnosisCoding'] += 1
    xys_c_1 = xys_c_0.loc[(xys_c_0["Right_Intraocular_Pressure"] > 10) & (xys_c_0["Left_Intraocular_Pressure"] > 10)]
    xys_c_1.loc[xys_c_1[(xys_c_1["Right_Intraocular_Pressure"] <= 21) & (
            xys_c_1["Left_Intraocular_Pressure"] <= 21)].index, "MajorDiagnosisCoding"] = 0
    xys_c_end = xys_c_1
    # 保存
    xys_c_end.reset_index(drop=True)
    return xys_c_end

def DefineModel_textAndNumerics(oneshape: typing.List[int], numclasses: int) -> Model:
    # # 分叉 神经网络
    text_input = Input(shape=(1, oneshape[0]), dtype=tf.float32, name='text_input')
    conv1D_1 = layers.Conv1D(32, 5, padding='same', activation=tf.nn.relu, input_shape=(1, oneshape[0]), name='conv1D_1')(
        text_input)
    conv1D_2 = layers.Conv1D(64, 5, padding='same', activation=tf.nn.relu, name='conv1D_2')(conv1D_1)
    #maxPooling_1 = layers.MaxPooling1D(5)(conv1D_2)
    dropout_1 = layers.Dropout(0.25)(conv1D_2)

    numerics_input = Input(shape=(1, oneshape[1]), dtype=tf.float32, name='numerics_input')
    conv1D_3 = layers.Conv1D(32, 5, padding='same', activation=tf.nn.relu, input_shape=(1, oneshape[1]), name='conv1D_3')(
        numerics_input)
    conv1D_4 = layers.Conv1D(64, 5, padding='same', activation=tf.nn.relu, name='conv1D_4')(conv1D_3)
    #maxPooling_2 = layers.MaxPooling1D(5)(conv1D_4)
    dropout_2 = layers.Dropout(0.25)(conv1D_4)

    concatenate_1 = layers.concatenate([dropout_1, dropout_2],axis=1)

    conv1D_5 = layers.Conv1D(128, 5, padding='same', activation=tf.nn.relu, input_shape=(oneshape[0] + oneshape[1], 1))(concatenate_1)
    conv1D_6 = layers.Conv1D(64, 5, padding='same', activation=tf.nn.relu)(conv1D_5)

    # maxPooling_3 = layers.MaxPooling1D(5)(conv1D_6)
    dropout_3 = layers.Dropout(0.25)(conv1D_6)
    conv1D_7 = layers.Conv1D(64, 5, padding='same', activation=tf.nn.relu)(dropout_3)
    globalAveragePooling1D_1 = layers.GlobalAveragePooling1D()(conv1D_7)
    dropout_4 = layers.Dropout(0.5)(globalAveragePooling1D_1)
    main_output = layers.Dense(numclasses, activation=tf.nn.softmax)(dropout_4)

    model = Model(inputs=[text_input, numerics_input], outputs=[main_output])

    return model

if __name__ == '__main__':
    main()
