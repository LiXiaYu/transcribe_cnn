import typing
import time
import os
import sys
import re

import numpy as np
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
import cnn_common as Kk
from cnn_vgg16x import cnn_vgg16x


class cnn_vgg16x_ac_combined(cnn_vgg16x):
    def __init__(self):
        super().__init__()
        self.py_filename = "cnn_vgg16x_ac_combined"
        self.CreateDir()

    def DefineModel_textAndNumerics_VGG16x(self, oneshape: typing.List[int], numclasses: int, ac_shape: int) -> Model:
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

        dropout_2 = layers.Dropout(0.25, name='dropout_2')(conv1D_4)

        concatenate_1 = layers.concatenate([dropout_1, dropout_2], axis=1)

        bn_1 = layers.BatchNormalization(axis=1)(concatenate_1)

        conv1D_vgg_1_1 = layers.Conv1D(64, 3, padding='same', activation=tf.nn.relu)(bn_1)
        conv1D_vgg_1_2 = layers.Conv1D(64, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_1_1)

        conv1D_vgg_2_1 = layers.Conv1D(128, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_1_2)
        conv1D_vgg_2_2 = layers.Conv1D(128, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_2_1)

        conv1D_vgg_3_1 = layers.Conv1D(256, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_2_2)
        conv1D_vgg_3_2 = layers.Conv1D(256, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_3_1)
        conv1D_vgg_3_3 = layers.Conv1D(256, 3, padding='same', activation=tf.nn.relu)(conv1D_vgg_3_2)

        flatten_vgg_6_1 = layers.Flatten()(conv1D_vgg_3_3)

        dense_vgg_6_4 = layers.Dense(numclasses, activation=tf.nn.softmax)(flatten_vgg_6_1)

        model = Model(inputs=[text_input, numerics_input], outputs=[dense_vgg_6_4])

        flatten_ac_1 = layers.Flatten()(dropout_2)

        dense_ac_end = layers.Dense(ac_shape, activation=tf.nn.softmax)(flatten_ac_1)

        model_ac = Model(inputs=[numerics_input], outputs=[dense_ac_end])
        self.model_ac = model_ac
        self.model = model
        return

    def TrainNet(self, data_process: typing.Callable[[np.array, np.array], typing.Tuple[np.array, np.array]] = None):

        self.SetAcResult(data_process)

        numclasses = self.y_train.max() + 1

        # 创建神经网络
        print("搭建神经网络")
        self.DefineModel_textAndNumerics_VGG16x([self.x_train_text.shape[1], self.x_train_numeric.shape[1]],
                                                numclasses, self.x_train_numeric_ld.shape[1])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model_ac.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                              loss='mean_squared_error',
                              metrics=['accuracy'])
        print(self.model.summary())
        print(self.model_ac.summary())
        print("训练神经网络")

        history_ac = self.model_ac.fit(self.x_train_numeric, self.x_train_numeric_ld, batch_size=200,
                                       validation_data=(self.x_test_numeric, self.x_test_numeric_ld), epochs=600)
        self.history_ac = history_ac

        metrics1 = Kc.Mertics_f1s_recalls_precisions(
            validation_data=([self.x_test_text, self.x_test_numeric], self.y_test))
        # metrics2 = Kc.Mertics_roc_auc(validation_data=([x_test_text, x_test_numeric], y_test))
        history = self.model.fit([self.x_train_text, self.x_train_numeric], self.y_train, batch_size=200,
                                 validation_data=([self.x_test_text, self.x_test_numeric], self.y_test), epochs=300,
                                 callbacks=[metrics1])
        y_test_result = self.model.predict([self.x_test_text, self.x_test_numeric])
        self.model = self.model
        self.metrics1 = metrics1
        self.history = history
        self.y_test_result = y_test_result
        return

    def Run(self, index_load: str = "false",
            data_process: typing.Callable[[np.array, np.array], typing.Tuple[np.array, np.array]] = None):
        self.PreprocessData_Load()

        self.SplitData(index_load=index_load)

        self.SeparateData()

        self.TrainNet(data_process=data_process)

        self.PostprocessData()

        return

    def PostprocessData(self):
        super().PostprocessData()

        try:
            plt.plot(self.history_ac.history['accuracy'])
            plt.plot(self.history_ac.history['val_accuracy'])

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
        return

    def SetAcResult(self, data_process: typing.Callable[[np.array, np.array], typing.Tuple[np.array, np.array]]):
        self.x_train_numeric_ld, self.x_test_numeric_ld = data_process(self.x_train_numeric, self.x_test_numeric)


def main():
    ca = cnn_vgg16x_ac_combined()

    # ca.py_filename = "cnn_vgg16x_ac_combined:TLCPD"
    # ca.CreateDir()
    # ca.Run(index_load="true", data_process=TLCPD)

    # ca.py_filename = "cnn_vgg16x_ac_combined:ICP"
    # ca.CreateDir()
    # ca.Run(index_load="true", data_process=ICP)

    # ca.py_filename = "cnn_vgg16x_ac_combined:V2"
    # ca.CreateDir()
    # ca.Run(index_load="true", data_process=V2)

    # ca.py_filename = "cnn_vgg16x_ac_combined:NoChange"
    # ca.CreateDir()
    # ca.Run(index_load="true", data_process=NoChange)

    ca.py_filename = "cnn_vgg16x_ac_combined:Fpr"
    ca.CreateDir()
    ca.Run(index_load="true", data_process=Fpr)
    # ca.SplitData_Save()
    return 0

def Fpr(x_train_numeric: np.array, x_test_numeric: np.array) -> typing.Tuple[np.array, np.array]:
    x_train_numeric_ld = x_train_numeric.reshape(x_train_numeric.shape[0], x_train_numeric.shape[1])
    x_test_numeric_ld = x_test_numeric.reshape(x_test_numeric.shape[0], x_test_numeric.shape[1])
    train_tlcpd_r = x_train_numeric_ld[:, 11] - (
            0.44 * x_train_numeric_ld[:, 7] / x_train_numeric_ld[:, 8] / x_train_numeric_ld[:,
                                                                         8] + 0.16 * x_train_numeric_ld[:,
                                                                                     6] - 0.18 * x_train_numeric_ld[
                                                                                                 :, 2] - 1.91)
    train_tlcpd_l = x_train_numeric_ld[:, 12] - (
            0.44 * x_train_numeric_ld[:, 7] / x_train_numeric_ld[:, 8] / x_train_numeric_ld[:,
                                                                         8] + 0.16 * x_train_numeric_ld[:,
                                                                                     6] - 0.18 * x_train_numeric_ld[
                                                                                                 :, 2] - 1.91)
    train_tlcpd_l[np.isnan(train_tlcpd_l)] = np.nanmean(train_tlcpd_l)
    train_tlcpd_r[np.isnan(train_tlcpd_r)] = np.nanmean(train_tlcpd_r)
    x_train_numeric_ld = np.insert(x_train_numeric_ld, 13, values=train_tlcpd_r, axis=1)
    x_train_numeric_ld = np.insert(x_train_numeric_ld, 14, values=train_tlcpd_l, axis=1)

    test_tlcpd_r = x_test_numeric_ld[:, 11] / (
            0.44 * x_test_numeric_ld[:, 7] / x_test_numeric_ld[:, 8] / x_test_numeric_ld[:,
                                                                       8] + 0.16 * x_test_numeric_ld[:,
                                                                                   6] - 0.18 * x_test_numeric_ld[
                                                                                               :, 2] - 1.91)
    test_tlcpd_l = x_test_numeric_ld[:, 12] / (
            0.44 * x_test_numeric_ld[:, 7] / x_test_numeric_ld[:, 8] / x_test_numeric_ld[:,
                                                                       8] + 0.16 * x_test_numeric_ld[:,
                                                                                   6] - 0.18 * x_test_numeric_ld[
                                                                                               :, 2] - 1.91)
    test_tlcpd_l[np.isnan(test_tlcpd_l)] = np.nanmean(test_tlcpd_l)
    test_tlcpd_r[np.isnan(test_tlcpd_r)] = np.nanmean(test_tlcpd_r)
    x_test_numeric_ld = np.insert(x_test_numeric_ld, 13, values=test_tlcpd_r, axis=1)
    x_test_numeric_ld = np.insert(x_test_numeric_ld, 14, values=test_tlcpd_l, axis=1)

    return x_train_numeric_ld, x_test_numeric_ld


def TLCPD(x_train_numeric: np.array, x_test_numeric: np.array) -> typing.Tuple[np.array, np.array]:
    x_train_numeric_ld = x_train_numeric.reshape(x_train_numeric.shape[0], x_train_numeric.shape[1])
    x_test_numeric_ld = x_test_numeric.reshape(x_test_numeric.shape[0], x_test_numeric.shape[1])
    train_tlcpd_r = x_train_numeric_ld[:, 11] - (
            0.44 * x_train_numeric_ld[:, 7] / x_train_numeric_ld[:, 8] / x_train_numeric_ld[:,
                                                                         8] + 0.16 * x_train_numeric_ld[:,
                                                                                     6] - 0.18 * x_train_numeric_ld[
                                                                                                 :, 2] - 1.91)
    train_tlcpd_l = x_train_numeric_ld[:, 12] - (
            0.44 * x_train_numeric_ld[:, 7] / x_train_numeric_ld[:, 8] / x_train_numeric_ld[:,
                                                                         8] + 0.16 * x_train_numeric_ld[:,
                                                                                     6] - 0.18 * x_train_numeric_ld[
                                                                                                 :, 2] - 1.91)
    train_tlcpd_l[np.isnan(train_tlcpd_l)] = np.nanmean(train_tlcpd_l)
    train_tlcpd_r[np.isnan(train_tlcpd_r)] = np.nanmean(train_tlcpd_r)
    x_train_numeric_ld = np.insert(x_train_numeric_ld, 13, values=train_tlcpd_r, axis=1)
    x_train_numeric_ld = np.insert(x_train_numeric_ld, 14, values=train_tlcpd_l, axis=1)

    test_tlcpd_r = x_test_numeric_ld[:, 11] - (
            0.44 * x_test_numeric_ld[:, 7] / x_test_numeric_ld[:, 8] / x_test_numeric_ld[:,
                                                                       8] + 0.16 * x_test_numeric_ld[:,
                                                                                   6] - 0.18 * x_test_numeric_ld[
                                                                                               :, 2] - 1.91)
    test_tlcpd_l = x_test_numeric_ld[:, 12] - (
            0.44 * x_test_numeric_ld[:, 7] / x_test_numeric_ld[:, 8] / x_test_numeric_ld[:,
                                                                       8] + 0.16 * x_test_numeric_ld[:,
                                                                                   6] - 0.18 * x_test_numeric_ld[
                                                                                               :, 2] - 1.91)
    test_tlcpd_l[np.isnan(test_tlcpd_l)] = np.nanmean(test_tlcpd_l)
    test_tlcpd_r[np.isnan(test_tlcpd_r)] = np.nanmean(test_tlcpd_r)
    x_test_numeric_ld = np.insert(x_test_numeric_ld, 13, values=test_tlcpd_r, axis=1)
    x_test_numeric_ld = np.insert(x_test_numeric_ld, 14, values=test_tlcpd_l, axis=1)

    return x_train_numeric_ld, x_test_numeric_ld


def ICP(x_train_numeric: np.array, x_test_numeric: np.array) -> typing.Tuple[np.array, np.array]:
    x_train_numeric_ld = x_train_numeric.reshape(x_train_numeric.shape[0], x_train_numeric.shape[1])
    x_test_numeric_ld = x_test_numeric.reshape(x_test_numeric.shape[0], x_test_numeric.shape[1])
    train_icp_r = (
            0.44 * x_train_numeric_ld[:, 7] / x_train_numeric_ld[:, 8] / x_train_numeric_ld[:,
                                                                         8] + 0.16 * x_train_numeric_ld[:,
                                                                                     6] - 0.18 * x_train_numeric_ld[
                                                                                                 :, 2] - 1.91)
    train_icp_l = (
            0.44 * x_train_numeric_ld[:, 7] / x_train_numeric_ld[:, 8] / x_train_numeric_ld[:,
                                                                         8] + 0.16 * x_train_numeric_ld[:,
                                                                                     6] - 0.18 * x_train_numeric_ld[
                                                                                                 :, 2] - 1.91)
    train_icp_l[np.isnan(train_icp_l)] = np.nanmean(train_icp_l)
    train_icp_r[np.isnan(train_icp_r)] = np.nanmean(train_icp_r)
    x_train_numeric_ld = np.insert(x_train_numeric_ld, 13, values=train_icp_r, axis=1)
    x_train_numeric_ld = np.insert(x_train_numeric_ld, 14, values=train_icp_l, axis=1)

    test_icp_r = (
            0.44 * x_test_numeric_ld[:, 7] / x_test_numeric_ld[:, 8] / x_test_numeric_ld[:,
                                                                       8] + 0.16 * x_test_numeric_ld[:,
                                                                                   6] - 0.18 * x_test_numeric_ld[
                                                                                               :, 2] - 1.91)
    test_icp_l = (
            0.44 * x_test_numeric_ld[:, 7] / x_test_numeric_ld[:, 8] / x_test_numeric_ld[:,
                                                                       8] + 0.16 * x_test_numeric_ld[:,
                                                                                   6] - 0.18 * x_test_numeric_ld[
                                                                                               :, 2] - 1.91)
    test_icp_l[np.isnan(test_icp_l)] = np.nanmean(test_icp_l)
    test_icp_r[np.isnan(test_icp_r)] = np.nanmean(test_icp_r)
    x_test_numeric_ld = np.insert(x_test_numeric_ld, 13, values=test_icp_r, axis=1)
    x_test_numeric_ld = np.insert(x_test_numeric_ld, 14, values=test_icp_l, axis=1)

    return x_train_numeric_ld, x_test_numeric_ld


def IOP(x_train_numeric: np.array, x_test_numeric: np.array) -> typing.Tuple[np.array, np.array]:
    x_train_numeric_ld = x_train_numeric.reshape(x_train_numeric.shape[0], x_train_numeric.shape[1])
    x_test_numeric_ld = x_test_numeric.reshape(x_test_numeric.shape[0], x_test_numeric.shape[1])
    train_iop_r = x_train_numeric_ld[:, 11]
    train_iop_l = x_train_numeric_ld[:, 12]
    train_iop_l[np.isnan(train_iop_l)] = np.nanmean(train_iop_l)
    train_iop_r[np.isnan(train_iop_r)] = np.nanmean(train_iop_r)
    x_train_numeric_ld = np.insert(x_train_numeric_ld, 13, values=train_iop_r, axis=1)
    x_train_numeric_ld = np.insert(x_train_numeric_ld, 14, values=train_iop_l, axis=1)

    test_iop_r = x_test_numeric_ld[:, 11]
    test_iop_l = x_test_numeric_ld[:, 12]
    test_iop_l[np.isnan(test_iop_l)] = np.nanmean(test_iop_l)
    test_iop_r[np.isnan(test_iop_r)] = np.nanmean(test_iop_r)
    x_test_numeric_ld = np.insert(x_test_numeric_ld, 13, values=test_iop_r, axis=1)
    x_test_numeric_ld = np.insert(x_test_numeric_ld, 14, values=test_iop_l, axis=1)

    return x_train_numeric_ld, x_test_numeric_ld


def V2(x_train_numeric: np.array, x_test_numeric: np.array) -> typing.Tuple[np.array, np.array]:
    x_train_numeric_ld = x_train_numeric.reshape(x_train_numeric.shape[0], x_train_numeric.shape[1])
    x_test_numeric_ld = x_test_numeric.reshape(x_test_numeric.shape[0], x_test_numeric.shape[1])

    train_v1 = x_train_numeric_ld[:, 2]
    train_v1[np.isnan(train_v1)] = np.nanmean(train_v1)
    x_train_numeric_ld = np.insert(x_train_numeric_ld, 12, values=train_v1, axis=1)

    test_v2 = x_test_numeric_ld[:, 2]
    test_v2[np.isnan(test_v2)] = np.nanmean(test_v2)
    x_test_numeric_ld = np.insert(x_test_numeric_ld, 13, values=test_v2, axis=1)

    return x_train_numeric_ld, x_test_numeric_ld


def NoChange(x_train_numeric: np.array, x_test_numeric: np.array) -> typing.Tuple[np.array, np.array]:
    x_train_numeric_ld = x_train_numeric.reshape(x_train_numeric.shape[0], x_train_numeric.shape[1])
    x_test_numeric_ld = x_test_numeric.reshape(x_test_numeric.shape[0], x_test_numeric.shape[1])

    return x_train_numeric_ld, x_test_numeric_ld


if __name__ == '__main__':
    main()
