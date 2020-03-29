import typing
import time
import os
import re
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
import matplotlib.pyplot as plt

o_path = os.getcwd()  # 返回当前工作目录
print(o_path)
sys.path.append(o_path)  # 添加自己指定的搜索路径
sys.path.append(os.path.dirname(os.path.dirname(__name__)))  # 上层路径的上层路径
import transcribe_common as Kc
# import cnn_common as Kk
import ReadDatas
from cnn_vgg16x import cnn_vgg16x


class beijing_cnn(cnn_vgg16x.cnn_vgg16x):
    def __init__(self):
        super().__init__()
        self.path_net = os.path.join('/', 'public', 'home', 'liqi', 'data', 'analysis', 'beijing_CNN')
        self.py_filename = "beijing_cnn"
        self.CreateDir()

    def PreprocessData(self):
        data = ReadDatas.ReadBeijingFromMySqlToXY(
            ["f_GENDER", "f_BIRTH", "f_BPPUL1", "f_BPSYS1", "f_BPDIA1", "f_WTKG", "f_HTCM", "f_IOPNCTR1", "f_IOPNCTL1"],
            "f_HISGLAUC")

        data = data[~data['f_GENDER'].isin([None])]

        data['f_BIRTH'] = data['f_BIRTH'].apply(lambda x: 2004 - x.year)
        data.dropna(axis=0, how='all')
        data.dropna(axis=1, how='all')

        # 数据插补
        print("开始插补数据")
        # #多重插补
        xys_mi = Kc.Interpolation_mice(data)
        print("插补数据完成")
        ############

        xys_mi.to_csv("Beijing_data9_mice.csv")

    def PreprocessData_Load(self, file_name: str = ""):
        if file_name == "":
            file_name = r'Beijing_data9_mice.csv'

        xys_c_end = pd.read_csv(os.path.join(self.path_net, file_name))
        xys_c_end.drop(xys_c_end.columns[[0]], axis=1, inplace=True)  # 删除多出来的序号列

        ###########
        self.xys_c_end = xys_c_end
        return xys_c_end

    def PreSet(self, preset_process: typing.Callable[[np.array], np.array] = None):
        if preset_process is None:
            return
        else:
            self.xys_c_end = preset_process(self.xys_c_end)
        return

    def SplitData(self, index_load: str = "false", train_name: str = 'train_index.npy',
                  test_name: str = 'test_index.npy'):
        xys_dd = self.xys_c_end.values

        if index_load == "false":
            self.train_index_list = []
            self.test_index_list = []
            x_train, y_train, x_test, y_test = Kc.SplitGroup(xys_dd, train_index_list=self.train_index_list,
                                                             test_index_list=self.test_index_list)
        elif index_load == "true":
            self.SplitData_Load(train_name, test_name)
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
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)
        return

    def DefineModel_VGG16x(self, oneshape: int, numclasses: int, ac_shape: int) -> Model:
        numeric_input = Input(shape=(oneshape, 1), dtype=tf.float32, name='numerics_input')

        numerics_bn = layers.BatchNormalization(axis=1)(numeric_input)

        conv1D_3 = layers.Conv1D(32, 5, padding='same', activation=tf.nn.relu, name='conv1D_3')(
            numerics_bn)
        conv1D_4 = layers.Conv1D(64, 5, padding='same', activation=tf.nn.relu, name='conv1D_4')(conv1D_3)
        # maxPooling_2 = layers.MaxPooling1D(5)(conv1D_4)
        dropout_2 = layers.Dropout(0.25, name='dropout_2')(conv1D_4)

        bn_1 = layers.BatchNormalization(axis=1)(dropout_2)

        conv1D_vgg_1_1 = layers.Conv1D(64, 3, padding='same', activation=tf.nn.relu)(bn_1)
        conv1D_vgg_1_2 = layers.Conv1D(64, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_1_1)
        # maxpooling1D_vgg_1_3 = layers.MaxPooling1D(3)(conv1D_vgg_1_2)

        conv1D_vgg_2_1 = layers.Conv1D(128, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_1_2)
        conv1D_vgg_2_2 = layers.Conv1D(128, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_2_1)
        # maxpooling1D_vgg_2_3 = layers.MaxPooling1D(3)(conv1D_vgg_2_2)

        conv1D_vgg_3_1 = layers.Conv1D(256, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_2_2)
        conv1D_vgg_3_2 = layers.Conv1D(256, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_3_1)
        conv1D_vgg_3_3 = layers.Conv1D(256, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_3_2)

        flatten_vgg_6_1 = layers.Flatten()(conv1D_vgg_3_3)

        dense_vgg_6_4 = layers.Dense(numclasses, activation=tf.nn.softmax)(flatten_vgg_6_1)

        model = Model(inputs=[numeric_input], outputs=[dense_vgg_6_4])

        flatten_ac_1 = layers.Flatten()(dropout_2)

        dense_ac_end = layers.Dense(ac_shape, activation=tf.nn.softmax)(flatten_ac_1)

        model_ac = Model(inputs=[numeric_input], outputs=[dense_ac_end])

        self.model = model
        self.model_ac = model_ac
        return

    def SetAcResult(self, data_process: typing.Callable[[np.array, np.array], typing.Tuple[np.array, np.array]] = None):
        if data_process is None:
            return
        else:
            self.x_train_ld, self.x_test_ld = data_process(self.x_train, self.x_test)
        return

    def TrainNet(self):
        numclasses = self.y_train.max() + 1
        # 创建神经网络
        print("搭建神经网络")

        if hasattr(self, "x_train_ld"):
            self.DefineModel_VGG16x(self.x_train.shape[1], numclasses, self.x_train_ld.shape[1])
            self.model_ac.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                                  loss='mean_squared_error',
                                  metrics=['accuracy'])
            print(self.model_ac.summary())

            history_ac = self.model_ac.fit(self.x_train, self.x_train_ld, batch_size=200,
                                           validation_data=(self.x_test, self.x_test_ld), epochs=600)
            self.history_ac = history_ac

        else:
            self.DefineModel_VGG16x(self.x_train.shape[1], numclasses, self.x_train.shape[1])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        print(self.model.summary())

        print("训练神经网络")

        metrics1 = Kc.Mertics_f1s_recalls_precisions(
            validation_data=(self.x_test, self.y_test))
        # metrics2 = Kc.Mertics_roc_auc(validation_data=([x_test_text, x_test_numeric], y_test))
        history = self.model.fit(self.x_train, self.y_train, batch_size=200,
                                 validation_data=(self.x_test, self.y_test), epochs=600,
                                 callbacks=[metrics1])
        y_test_result = self.model.predict(self.x_test)
        self.model = self.model
        self.metrics1 = metrics1
        self.history = history
        self.y_test_result = y_test_result

        return

    def PostprocessData(self):
        try:
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.plot(self.metrics1.val_f1s)
            plt.plot(self.metrics1.val_recalls)
            plt.plot(self.metrics1.val_precisions)
            plt.legend(['training', 'validation', 'val_f1', 'val_recall', 'val_precision'], loc='lower right')
            plt.title(self.py_filename)
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
                                 'x_test.npy'),
                    self.x_test)
            np.save(os.path.join(self.path_net, 'result', self.py_filename, self.nowtime + r'_' + self.py_filename,
                                 'y_test_result.npy'),
                    self.y_test_result)
            np.save(os.path.join(self.path_net, 'result', self.py_filename, self.nowtime + r'_' + self.py_filename,
                                 'y_test.npy'), self.y_test)

        if hasattr(self, "x_train_ld"):
            try:
                plt.plot(self.history_ac.history['accuracy'])
                plt.plot(self.history_ac.history['val_accuracy'])
                plt.title(self.py_filename + "_ac")
                plt.legend(['training', 'validation'], loc='lower right')
                plt.savefig(
                    os.path.join(self.path_net, 'result', self.py_filename, self.nowtime + '_cnn_ac_result' + '.png'))
                plt.ion()
                plt.show()
                plt.close()
            except:
                print("无法画图")
            finally:
                with open(os.path.join(self.path_net, 'result', self.py_filename, self.nowtime + '_history_ac.txt'),
                          'w+') as f:
                    f.write(str(self.history_ac.history))
                self.model.save(os.path.join(self.path_net, 'model', self.py_filename,
                                             self.nowtime + r'_' + self.py_filename + "_ac" + '.model'))
        else:
            return

        return

    def Run(self, file_name: str = "",
            index_load: str = "false", train_name: str = 'train_index.npy', test_name: str = 'test_index.npy',
            preset_process: typing.Callable[[np.array], np.array] = None,
            data_process: typing.Callable[[np.array, np.array], typing.Tuple[np.array, np.array]] = None):
        self.PreprocessData_Load(file_name=file_name)

        self.PreSet(preset_process=preset_process)

        self.SplitData(index_load=index_load, train_name=train_name, test_name=test_name)

        self.SeparateData()

        self.SetAcResult(data_process)

        self.TrainNet()

        self.PostprocessData()

        return

    def SplitData_Save(self, train_name: str = 'train_index.npy', test_name: str = 'test_index.npy'):
        np.save(os.path.join(self.path_net, train_name), np.array(self.train_index_list))
        np.save(os.path.join(self.path_net, test_name), np.array(self.test_index_list))

    def SplitData_Load(self, train_name: str = 'train_index.npy', test_name: str = 'test_index.npy'):
        self.train_index_list = np.load(os.path.join(self.path_net, train_name)).tolist()[0]
        self.test_index_list = np.load(os.path.join(self.path_net, test_name)).tolist()[0]


def main():
    # bc = beijing_cnn()
    # bc.py_filename = "beijing_cnn:preset_tlcpd"
    # bc.CreateDir()
    # bc.Run(index_load="true", preset_process=preset_tlcpd)
    #
    # bc = beijing_cnn()
    # bc.py_filename = "beijing_cnn:ac_none"
    # bc.CreateDir()
    # bc.Run(index_load="true", data_process=ac_NoChange)
    #
    # bc = beijing_cnn()
    # bc.py_filename = "beijing_cnn:ac_tlcpd"
    # bc.CreateDir()
    # bc.Run(index_load="true", data_process=ac_TLCPD)

    # bc = beijing_cnn()
    # bc.py_filename = "beijing_cnn:numeric:none:none"
    # bc.CreateDir()
    # bc.Run(file_name="Beijing_data_numeric_nanmean.csv", index_load="true")
    #
    # bc = beijing_cnn()
    # bc.py_filename = "beijing_cnn:numeric:preset_tlcpd:none"
    # bc.CreateDir()
    # bc.Run(file_name="Beijing_data_numeric_nanmean.csv", index_load="true", preset_process=preset_tlcpd)
    #
    # bc = beijing_cnn()
    # bc.py_filename = "beijing_cnn:numeric:none:noChange"
    # bc.CreateDir()
    # bc.Run(file_name="Beijing_data_numeric_nanmean.csv", index_load="true", data_process=ac_NoChange)

    # bc = beijing_cnn()
    # bc.py_filename = "beijing_cnn:numeric:none:TLCPD"
    # bc.CreateDir()
    # bc.Run(file_name="Beijing_data_numeric_nanmean.csv", index_load="true", data_process=ac_TLCPD)
    bc = beijing_cnn()
    bc.py_filename = "beijing_cnn:numeric:preset_fpr:none"
    bc.CreateDir()
    bc.Run(file_name="Beijing_data_numeric_nanmean.csv", index_load="true", preset_process=preset_fpr)

    bc = beijing_cnn()
    bc.py_filename = "beijing_cnn:numeric:none:FPR"
    bc.CreateDir()
    bc.Run(file_name="Beijing_data_numeric_nanmean.csv", index_load="true", data_process=ac_FPR)

    return 0


def preset_tlcpd(data):
    iop_r = data['f_IOPNCTR1']
    iop_l = data['f_IOPNCTL1']

    icp = 0.44 * data['f_WTKG'] / data['f_HTCM'] / data['f_HTCM'] + 0.16 * data['f_BPDIA1'] - 0.18 * \
          data['year'] - 1.91

    tlcpd_r = iop_r - icp
    tlcpd_l = iop_l - icp
    data.insert(data.shape[1], 'Left_TLCPD', tlcpd_l)
    data.insert(data.shape[1], 'Right_TLCPD', tlcpd_r)
    data = data.replace([np.inf, -np.inf], np.nan)

    # data['TLCPD'] = tlcpd_re
    return data.fillna(data.mean())

def preset_fpr(data):
    iop_r = data['f_IOPNCTR1']
    iop_l = data['f_IOPNCTL1']

    icp = 0.44 * data['f_WTKG'] / data['f_HTCM'] / data['f_HTCM'] + 0.16 * data['f_BPDIA1'] - 0.18 * \
          data['year'] - 1.91

    fpr_r = iop_r / icp
    fpr_l = iop_l / icp
    data.insert(data.shape[1], 'Left_FPR', fpr_l)
    data.insert(data.shape[1], 'Right_FPR', fpr_r)
    data = data.replace([np.inf, -np.inf], np.nan)

    # data['TLCPD'] = tlcpd_re
    return data.fillna(data.mean())

def ac_NoChange(x_train: np.array, x_test: np.array) -> typing.Tuple[np.array, np.array]:
    x_train_ld = x_train.reshape(x_train.shape[0], x_train.shape[1])
    x_test_ld = x_test.reshape(x_test.shape[0], x_test.shape[1])

    return x_train_ld, x_test_ld


def ac_TLCPD(x_train: np.array, x_test: np.array) -> typing.Tuple[np.array, np.array]:
    x_train_ld = x_train.reshape(x_train.shape[0], x_train.shape[1])
    x_test_ld = x_test.reshape(x_test.shape[0], x_test.shape[1])
    train_tlcpd_r = x_train_ld[:, 304] - (
            0.44 * x_train_ld[:, 40] / x_train_ld[:, 38] / x_train_ld[:,
                                                           38] + 0.16 * x_train_ld[:,
                                                                        32] - 0.18 * x_train_ld[
                                                                                     :, 490] - 1.91)
    train_tlcpd_l = x_train_ld[:, 305] - (
            0.44 * x_train_ld[:, 40] / x_train_ld[:, 38] / x_train_ld[:,
                                                           38] + 0.16 * x_train_ld[:,
                                                                        32] - 0.18 * x_train_ld[
                                                                                     :, 490] - 1.91)
    train_tlcpd_l[np.isnan(train_tlcpd_l)] = np.nanmean(train_tlcpd_l)
    train_tlcpd_r[np.isnan(train_tlcpd_r)] = np.nanmean(train_tlcpd_r)
    x_train_ld = np.insert(x_train_ld, x_train_ld.shape[1], values=train_tlcpd_r, axis=1)
    x_train_ld = np.insert(x_train_ld, x_train_ld.shape[1], values=train_tlcpd_l, axis=1)

    test_tlcpd_r = x_test_ld[:, 304] - (
            0.44 * x_test_ld[:, 40] / x_test_ld[:, 38] / x_test_ld[:,
                                                         38] + 0.16 * x_test_ld[:,
                                                                      32] - 0.18 * x_test_ld[
                                                                                   :, 490] - 1.91)
    test_tlcpd_l = x_test_ld[:, 305] - (
            0.44 * x_test_ld[:, 40] / x_test_ld[:, 38] / x_test_ld[:,
                                                         38] + 0.16 * x_test_ld[:,
                                                                      32] - 0.18 * x_test_ld[
                                                                                   :, 490] - 1.91)

    test_tlcpd_l[np.isnan(test_tlcpd_l)] = np.nanmean(test_tlcpd_l)
    test_tlcpd_r[np.isnan(test_tlcpd_r)] = np.nanmean(test_tlcpd_r)
    x_test_ld = np.insert(x_test_ld, x_test_ld.shape[1], values=test_tlcpd_r, axis=1)
    x_test_ld = np.insert(x_test_ld, x_test_ld.shape[1], values=test_tlcpd_l, axis=1)

    return x_train_ld, x_test_ld

def ac_FPR(x_train: np.array, x_test: np.array) -> typing.Tuple[np.array, np.array]:
    x_train_ld = x_train.reshape(x_train.shape[0], x_train.shape[1])
    x_test_ld = x_test.reshape(x_test.shape[0], x_test.shape[1])
    train_fpr_r = x_train_ld[:, 304] / (
            0.44 * x_train_ld[:, 40] / x_train_ld[:, 38] / x_train_ld[:,
                                                           38] + 0.16 * x_train_ld[:,
                                                                        32] - 0.18 * x_train_ld[
                                                                                     :, 490] - 1.91)
    train_fpr_l = x_train_ld[:, 305] / (
            0.44 * x_train_ld[:, 40] / x_train_ld[:, 38] / x_train_ld[:,
                                                           38] + 0.16 * x_train_ld[:,
                                                                        32] - 0.18 * x_train_ld[
                                                                                     :, 490] - 1.91)
    train_fpr_l[np.isnan(train_fpr_l)] = np.nanmean(train_fpr_l)
    train_fpr_r[np.isnan(train_fpr_r)] = np.nanmean(train_fpr_r)
    x_train_ld = np.insert(x_train_ld, x_train_ld.shape[1], values=train_fpr_r, axis=1)
    x_train_ld = np.insert(x_train_ld, x_train_ld.shape[1], values=train_fpr_l, axis=1)

    test_fpr_r = x_test_ld[:, 304] - (
            0.44 * x_test_ld[:, 40] / x_test_ld[:, 38] / x_test_ld[:,
                                                         38] + 0.16 * x_test_ld[:,
                                                                      32] - 0.18 * x_test_ld[
                                                                                   :, 490] - 1.91)
    test_fpr_l = x_test_ld[:, 305] - (
            0.44 * x_test_ld[:, 40] / x_test_ld[:, 38] / x_test_ld[:,
                                                         38] + 0.16 * x_test_ld[:,
                                                                      32] - 0.18 * x_test_ld[
                                                                                   :, 490] - 1.91)

    test_fpr_l[np.isnan(test_fpr_l)] = np.nanmean(test_fpr_l)
    test_fpr_r[np.isnan(test_fpr_r)] = np.nanmean(test_fpr_r)
    x_test_ld = np.insert(x_test_ld, x_test_ld.shape[1], values=test_fpr_r, axis=1)
    x_test_ld = np.insert(x_test_ld, x_test_ld.shape[1], values=test_fpr_l, axis=1)

    return x_train_ld, x_test_ld

if __name__ == '__main__':
    main()
