import typing
import time
import os
import re

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
import cnn_common as Kk
from transcribe_common import Mertics_f1s_recalls_precisions


class cnn_vgg16x(Kk.cnn_common):

    def __init__(self):
        super().__init__()

        self.py_filename = "cnn_vgg16x"

        self.CreateDir()

    def PreprocessData(self):
        self.data = pd.read_csv(os.path.join(self.path_net, "save_CNN_text_2_datas.csv"))
        numeric_xys = Kc.DataFrameToXY(self.data,
                                       ["Sex", "Years", "Temperature", "Pulse", "Breath", "Blood_Pressure_Systolic",
                                        "Blood_Pressure_Diastolic", "Width", "Height", "Corneal_Endothelium_Right",
                                        "Corneal_Endothelium_Left", "Right_Intraocular_Pressure",
                                        "Left_Intraocular_Pressure"],
                                       "MajorDiagnosisCoding",
                                       manualFilePath=os.path.join(self.path_net, r"CodingToClass_1to16 - new 多分类.csv"))
        text_xys = Kc.DataFrameToXY(self.data, ["HistoryOfPastIllness"], "MajorDiagnosisCoding",
                                    manualFilePath=os.path.join(self.path_net, r"CodingToClass_1to16 - new 多分类.csv"))
        # # 排除不要的标记数据
        numeric_xys = numeric_xys[numeric_xys["MajorDiagnosisCoding"] != "-1"]
        text_xys = text_xys[text_xys["MajorDiagnosisCoding"] != "-1"]
        # 字符串向量化
        model, text_x_vds = Kc.TextToVector_doc2vec(text_xys["HistoryOfPastIllness"].values.tolist())
        model.save(os.path.join(self.path_net, r'model_text_2.model'))
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
        xys_c_1 = xys_c_0.loc[
            (xys_c_0["Right_Intraocular_Pressure"] > 10) & (xys_c_0["Left_Intraocular_Pressure"] > 10)]
        xys_c_1.loc[xys_c_1[(xys_c_1["Right_Intraocular_Pressure"] <= 21) & (
                xys_c_1["Left_Intraocular_Pressure"] <= 21)].index, "MajorDiagnosisCoding"] = 0
        xys_c_end = xys_c_1
        # 保存
        xys_c_end.reset_index(drop=True)
        self.xys_c_end = xys_c_end
        return xys_c_end

    def SplitData(self, index_load: str = "false"):
        xys_dd = self.xys_c_end.values

        if index_load == "false":
            self.train_index_list = []
            self.test_index_list = []
            x_train, y_train, x_test, y_test = Kc.SplitGroup(xys_dd, train_index_list=self.train_index_list,
                                                             test_index_list=self.test_index_list)
        elif index_load == "true":
            self.SplitData_Load()
            y_test = xys_dd[self.test_index_list, 0]
            x_test = xys_dd[self.test_index_list, 1:]
            y_train = xys_dd[self.train_index_list, 0]
            x_train = xys_dd[self.train_index_list, 1:]

        y_train = y_train.astype(np.float64)
        y_test = y_test.astype(np.float64)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        return x_train, y_train, x_test, y_test

    def SeparateData(self):
        # 分离数据
        x_train_text = np.array(list(map(lambda x: x[-1], self.x_train)))

        x_train_numeric = np.array(list(map(lambda x: x[0:-1], self.x_train)), dtype=np.float64)
        y_train_all = self.y_train
        x_test_text = np.array(list(map(lambda x: x[-1], self.x_test)))

        x_test_numeric = np.array(list(map(lambda x: x[0:-1], self.x_test)), dtype=np.float64)
        y_test_all = self.y_test

        x_train_text = x_train_text.reshape(x_train_text.shape[0], x_train_text.shape[1], 1)
        x_train_numeric = x_train_numeric.reshape(x_train_numeric.shape[0], x_train_numeric.shape[1], 1)
        x_test_text = x_test_text.reshape(x_test_text.shape[0], x_test_text.shape[1], 1)
        x_test_numeric = x_test_numeric.reshape(x_test_numeric.shape[0], x_test_numeric.shape[1], 1)

        self.x_train_text = x_train_text
        self.x_train_numeric = x_train_numeric
        self.x_test_text = x_test_text
        self.x_test_numeric = x_test_numeric
        self.y_train_all = y_train_all
        self.y_test_all = y_test_all
        return x_train_text, x_train_numeric, x_test_text, x_test_numeric, y_train_all, y_test_all

    def DefineModel_textAndNumerics_VGG16x(self, oneshape: typing.List[int], numclasses: int) -> Model:
        # # 分叉 神经网络
        text_input = Input(shape=(oneshape[0], 1), dtype=tf.float32, name='text_input')

        text_bn = layers.BatchNormalization(axis=1)(text_input)

        conv1D_1 = layers.Conv1D(32, 5, padding='same', activation=tf.nn.relu, name='conv1D_1')(
            text_bn)
        conv1D_2 = layers.Conv1D(64, 5, padding='same', activation=tf.nn.relu, name='conv1D_2')(conv1D_1)
        # maxPooling_1 = layers.MaxPooling1D(5)(conv1D_2)
        dropout_1 = layers.Dropout(0.25)(conv1D_2)

        numerics_input = Input(shape=(oneshape[1], 1), dtype=tf.float32, name='numerics_input')

        numerics_bn = layers.BatchNormalization(axis=1)(numerics_input)

        conv1D_3 = layers.Conv1D(32, 5, padding='same', activation=tf.nn.relu, name='conv1D_3')(
            numerics_bn)
        conv1D_4 = layers.Conv1D(64, 5, padding='same', activation=tf.nn.relu, name='conv1D_4')(conv1D_3)
        # maxPooling_2 = layers.MaxPooling1D(5)(conv1D_4)
        dropout_2 = layers.Dropout(0.25, name='dropout_2')(conv1D_4)

        concatenate_1 = layers.concatenate([dropout_1, dropout_2], axis=1)

        bn_1 = layers.BatchNormalization(axis=1)(concatenate_1)

        conv1D_vgg_1_1 = layers.Conv1D(64, 3, padding='same', activation=tf.nn.relu)(bn_1)
        conv1D_vgg_1_2 = layers.Conv1D(64, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_1_1)
        # maxpooling1D_vgg_1_3 = layers.MaxPooling1D(3)(conv1D_vgg_1_2)

        conv1D_vgg_2_1 = layers.Conv1D(128, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_1_2)
        conv1D_vgg_2_2 = layers.Conv1D(128, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_2_1)
        # maxpooling1D_vgg_2_3 = layers.MaxPooling1D(3)(conv1D_vgg_2_2)

        conv1D_vgg_3_1 = layers.Conv1D(256, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_2_2)
        conv1D_vgg_3_2 = layers.Conv1D(256, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_3_1)
        conv1D_vgg_3_3 = layers.Conv1D(256, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_3_2)
        # maxpooling1D_vgg_3_4 = layers.MaxPooling1D(3)(conv1D_vgg_3_3)

        # conv1D_vgg_4_1 = layers.Conv1D(512, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_3_3)
        # conv1D_vgg_4_2 = layers.Conv1D(512, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_4_1)
        # conv1D_vgg_4_3 = layers.Conv1D(512, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_4_2)
        # # maxpooling1D_vgg_4_4 = layers.MaxPooling1D(3)(conv1D_vgg_4_3)

        # conv1D_vgg_5_1 = layers.Conv1D(512, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_4_3)
        # conv1D_vgg_5_2 = layers.Conv1D(512, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_5_1)
        # conv1D_vgg_5_3 = layers.Conv1D(512, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_5_2)
        # maxpooling1D_vgg_5_4 = layers.MaxPooling1D(3)(conv1D_vgg_5_3)

        flatten_vgg_6_1 = layers.Flatten()(conv1D_vgg_3_3)
        # dense_vgg_6_2 = layers.Dense(int(2*(512+numclasses) / 3), activation=tf.nn.softmax)(flatten_vgg_6_1)
        # dense_vgg_6_3 = layers.Dense(int((512 + numclasses) / 3), activation=tf.nn.softmax)(dense_vgg_6_2)
        dense_vgg_6_4 = layers.Dense(numclasses, activation=tf.nn.softmax)(flatten_vgg_6_1)

        model = Model(inputs=[text_input, numerics_input], outputs=[dense_vgg_6_4])

        return model

    def PostprocessData(self):
        try:
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.plot(self.metrics1.val_f1s)
            plt.plot(self.metrics1.val_recalls)
            plt.plot(self.metrics1.val_precisions)
            plt.legend(['training', 'validation', 'val_f1', 'val_recall', 'val_precision'], loc='lower right')

            plt.savefig(os.path.join(self.path_net, 'result', self.py_filename, self.nowtime + '_cnn_result' + '.png'))
            plt.ion()
            plt.show()
            plt.close()

            # plt.plot(metrics2.val_fprs, metrics2.val_tprs)
            # plt.show()
        except:
            print("无法画图")
        finally:
            with open(os.path.join(self.path_net, 'result', self.py_filename, self.nowtime + '_history.txt'),
                      'w+') as f:
                f.write(str(self.history.history))
            self.model.save(os.path.join(self.path_net, 'model', self.py_filename,
                                         self.nowtime + r'_' + self.py_filename + '.model'))

            Kc.CreateDir(
                os.path.join(self.path_net, 'result', self.py_filename, self.nowtime + r'_' + self.py_filename))
            np.save(os.path.join(self.path_net, 'result', self.py_filename, self.nowtime + r'_' + self.py_filename,
                                 'x_test_text.npy'),
                    self.x_test_text)
            np.save(os.path.join(self.path_net, 'result', self.py_filename, self.nowtime + r'_' + self.py_filename,
                                 'x_test_numeric.npy'),
                    self.x_test_numeric)
            np.save(os.path.join(self.path_net, 'result', self.py_filename, self.nowtime + r'_' + self.py_filename,
                                 'y_test_result.npy'),
                    self.y_test_result)
            np.save(os.path.join(self.path_net, 'result', self.py_filename, self.nowtime + r'_' + self.py_filename,
                                 'y_test.npy'), self.y_test)

        return

    def TrainNet(self):
        numclasses = self.y_train.max() + 1
        # 创建神经网络
        print("搭建神经网络")
        model = self.DefineModel_textAndNumerics_VGG16x([self.x_train_text.shape[1], self.x_train_numeric.shape[1]],
                                                        numclasses)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())
        print("训练神经网络")
        metrics1 = Kc.Mertics_f1s_recalls_precisions(
            validation_data=([self.x_test_text, self.x_test_numeric], self.y_test))
        # metrics2 = Kc.Mertics_roc_auc(validation_data=([x_test_text, x_test_numeric], y_test))
        history = model.fit([self.x_train_text, self.x_train_numeric], self.y_train, batch_size=200,
                            validation_data=([self.x_test_text, self.x_test_numeric], self.y_test), epochs=300,
                            callbacks=[metrics1])
        y_test_result = model.predict([self.x_test_text, self.x_test_numeric])
        self.model = model
        self.metrics1 = metrics1
        self.history = history
        self.y_test_result = y_test_result

        return

    def Run(self, index_load: str = "false"):
        self.PreprocessData_Load()

        self.SplitData(index_load = index_load)

        self.SeparateData()

        self.TrainNet()

        self.PostprocessData()

        return

    def PreprocessData_Load(self):
        xys_c_end = pd.read_csv(os.path.join(self.path_net, r'xys_c_end.csv'))
        xys_c_end.drop(xys_c_end.columns[[0]], axis=1, inplace=True)  # 删除多出来的序号列
        xys_c_end['HistoryOfPastIllness_vec'] = xys_c_end['HistoryOfPastIllness_vec'].apply(
            lambda x: eval(re.sub(r'(?<=\d)\s+', ',', x, flags=re.S)))
        xys_c_end.loc[xys_c_end[xys_c_end['MajorDiagnosisCoding'] != 0].index, 'MajorDiagnosisCoding'] = 1
        ###########
        self.xys_c_end = xys_c_end
        return xys_c_end

    def SplitData_Save(self):
        np.save(os.path.join(self.path_net, 'train_index.npy'), np.array(self.train_index_list))
        np.save(os.path.join(self.path_net, 'test_index.npy'), np.array(self.test_index_list))

    def SplitData_Load(self):
        self.train_index_list = np.load(os.path.join(self.path_net, 'train_index.npy')).tolist()[0]
        self.test_index_list = np.load(os.path.join(self.path_net, 'test_index.npy')).tolist()[0]


def main():
    ca = cnn_vgg16x()
    ca.Run(index_load="true")
    # ca.SplitData_Save()
    return 0


if __name__ == '__main__':
    main()
