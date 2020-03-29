import typing
import os

import numpy as np
import random
import pandas as pd
import thulac
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score, precision_score, recall_score
import statsmodels.api as sm

import ReadDatas

# 病编码转换
# def MDCodetoIntCode(mdc):
#     mdtoic = []
#     ic = []
#     for md in mdc:
#         try:
#             x = mdtoic.index(md)
#         # except BaseException as err:
#         #    ee=1
#         except ValueError:
#             mdtoic.append(md)
#             x = len(mdtoic)
#         ic.append(x)
#     return mdtoic, ic


# 根据X和Y的名字取出数据来
def DataFrameToXY(dataFrame: pd.DataFrame, nameXs: object, nameY: object, manualFilePath: str = "") -> pd.DataFrame:
    y_str = dataFrame[nameY]
    mdtoic, y_data = MDCodetoIntCode(y_str.tolist(), manualFilePath)

    # 转置list，提取x
    x_data = list(map(list, zip(*list(map(lambda a: list(dataFrame[a]), nameXs)))))

    xys = np.hstack((np.array(y_data).reshape((len(y_data), 1)), np.array(x_data)))

    datas = pd.DataFrame(xys, columns=[nameY] + nameXs)

    return datas


def Interpolation_mice(df: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.imputation import mice
    imp = mice.MICEData(df)
    fml = df.columns[0] + " ~ " + df.columns[1]
    for i in range(2, len(df.columns)):
        fml += " + " + df.columns[i]
    # fml = 'y ~ x1 + x2 + x3 + x4'
    mi = mice.MICE(fml, sm.OLS, imp)
    results = mi.fit(10, 10)
    dm = imp.next_sample()

    # dm.to_csv("data_mice_10.csv")

    # results = []
    # for k in range(10):
    #    x = mi.next_sample()
    #    results.append(x)

    # TODO:
    # results到底怎么个结果？
    # FINISHED
    return dm


# 读取数据，并且多重线性插补
def ReadAndMice_sample1():
    datas = ReadDatas.ReadFromMySqlToDataFrame()
    xys = DataFrameToXY(datas, ["Sex", "Years", "Temperature", "Pulse", "Breath", "Blood_Pressure_Systolic",
                                "Blood_Pressure_Diastolic", "Width", "Height", "Corneal_Endothelium_Right",
                                "Corneal_Endothelium_Left", "Right_Intraocular_Pressure", "Left_Intraocular_Pressure"],
                        "MajorDiagnosisCoding")

    xys[list(xys.columns)] = xys[list(xys.columns)].apply(pd.to_numeric, errors='coerce')
    xys = xys.astype(float)
    # 删除缺项
    # xys_drop = xys.dropna().values
    # xys_drop = xys_drop.astype(float)
    ############

    # 多重插补
    # xys_df = pd.DataFrame(xys,columns=[keynames[i] for i in [5,3,4,8,9,10,11,12,13,14,15,16,17,18]]).astype(float)

    # 数据插补
    print("开始插补数据")
    # #多重插补
    xys_mi = Interpolation_mice(xys)
    print("插补数据完成")
    ############

    return xys_mi


def Mice_numeric(xys: pd.DataFrame) -> pd.DataFrame:
    xys[list(xys.columns)] = xys[list(xys.columns)].apply(pd.to_numeric, errors='coerce')
    xys = xys.astype(float)
    # 数据插补

    # #多重插补
    xys_mi = Interpolation_mice(xys)
    return xys_mi


# 病编码转换，手动转换手动分组
def MDCodetoIntCode_Manual(mdc: list) -> typing.Tuple[pd.DataFrame, list]:
    Cs = pd.read_csv(r'CodingToClass.csv')
    As = []
    Bs = []
    ic = []

    for md in mdc:
        md_mdc = Cs.loc[Cs['MajorDiagnosisCoding'] == md]
        x = md_mdc['MajorDiagnosisClass']
        ic.append(x.values[0])
    return Cs, ic


# 根据X和Y的名字取出数据来
def DataFrameToXY_Manual(dataFrame: pd.DataFrame, nameXs: list, nameY: str) -> pd.DataFrame:
    y_str = dataFrame[nameY]
    mdtoic, y_data = MDCodetoIntCode_Manual(y_str.tolist())

    # 转置list，提取x
    x_data = list(map(list, zip(*list(map(lambda a: list(dataFrame[a]), nameXs)))))

    xys = np.hstack((np.array(y_data).reshape((len(y_data), 1)), np.array(x_data)))

    datas = pd.DataFrame(xys, columns=[nameY] + nameXs)

    return datas


# 读取数据，并且多重线性插补
def ReadAndMice_sample2() -> pd.DataFrame:
    datas = ReadDatas.ReadFromMySqlToDataFrame()
    xys = DataFrameToXY(datas, ["Sex", "Years", "Temperature", "Pulse", "Breath", "Blood_Pressure_Systolic",
                                "Blood_Pressure_Diastolic", "Width", "Height", "Corneal_Endothelium_Right",
                                "Corneal_Endothelium_Left", "Right_Intraocular_Pressure",
                                "Left_Intraocular_Pressure"],
                        "MajorDiagnosisCoding",
                        manualFilePath=r"CodingToClass_1to16.csv")

    xys[list(xys.columns)] = xys[list(xys.columns)].apply(pd.to_numeric, errors='coerce')
    xys = xys.astype(float)
    # 删除缺项
    # xys_drop = xys.dropna().values
    # xys_drop = xys_drop.astype(float)
    ############

    # 多重插补
    # xys_df = pd.DataFrame(xys,columns=[keynames[i] for i in [5,3,4,8,9,10,11,12,13,14,15,16,17,18]]).astype(float)

    # 数据插补
    print("开始插补数据")
    # #多重插补
    xys_mi = Interpolation_mice(xys)
    print("插补数据完成")
    ############

    return xys_mi


def MDCodeTransformAsFile(mdc: list, path: str) -> typing.Tuple[pd.DataFrame, list]:
    Cs = pd.read_csv(path)
    As = []
    Bs = []
    ic = []

    for md in mdc:
        md_mdc = Cs.loc[Cs['MajorDiagnosisCoding'] == md]
        x = md_mdc['MajorDiagnosisClass']
        ic.append(x.values[0])
    return Cs, ic


# 分训练组和测试组
def SetGroupTrainAndTest(xys_dd: np.array) -> typing.Tuple[np.array, np.array, np.array, np.array]:
    # 分训练组和测试组
    test_r = 0.3
    index_list = list(range(0, len(xys_dd)))
    test_index = random.sample(index_list, int(len(xys_dd) * test_r))
    x_test = xys_dd[test_index, 2:]
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    y_test = xys_dd[test_index, 1]
    train_index = index_list.copy()
    # temp_daieo=map(lambda a,ti=train_index:ti.remove(a),test_index)
    for i in test_index:
        train_index.remove(i)

    x_train = xys_dd[train_index, 2:]
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    y_train = xys_dd[train_index, 1]

    return x_train, y_train, x_test, y_test
    # numclasses = y_train.max() + 1
    # onesharp = len(x_train[0])


def SplitGroup(xys: np.array) -> typing.Tuple[np.array, np.array, np.array, np.array]:
    # 分训练组和测试组
    test_r = 0.3
    index_list = list(range(0, len(xys)))
    test_index = random.sample(index_list, int(len(xys) * test_r))

    train_index = index_list.copy()
    for i in test_index:
        train_index.remove(i)

    y_test = xys[test_index, 0]
    x_test = xys[test_index, 1:]

    y_train = xys[train_index, 0]
    x_train = xys[train_index, 1:]

    return x_train, y_train, x_test, y_test


def SplitIndex(num: int) -> typing.Tuple[np.array, np.array]:
    # 分训练组和测试组
    test_r = 0.3
    index_list = list(range(0, num))
    test_index = random.sample(index_list, int(num * test_r))

    train_index = index_list.copy()
    for i in test_index:
        train_index.remove(i)

    return train_index, test_index


def Word2vec(text: str) -> np.array:
    thu1 = thulac.thulac()  # 默认模式
    text_la = thu1.cut(text, text=True)  # 进行一句话分词

    return np.array()


# 病编码转换，手动转换手动分组
def MDCodetoIntCode(mdc: list, manualFilePath: str = "") -> typing.Tuple[pd.DataFrame, list]:
    if manualFilePath == "":
        mdtoic = []
        ic = []
        for md in mdc:
            try:
                x = mdtoic.index(md)
            # except BaseException as err:
            #    ee=1
            except ValueError:
                mdtoic.append(md)
                x = len(mdtoic)
            ic.append(x)
        return mdtoic, ic
    else:
        Cs = pd.read_csv(manualFilePath)
        As = []
        Bs = []
        ic = []

        for md in mdc:
            md_mdc = Cs.loc[Cs['MajorDiagnosisCoding'] == md]
            x = md_mdc['MajorDiagnosisClass']
            ic.append(x.values[0])
        return Cs, ic


class Metrics(keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict_onehot = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_predict = np.array(list(map(np.argmax, val_predict_onehot)))  # onehot转正常
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average="macro")
        _val_recall = recall_score(val_targ, val_predict, average="macro")
        _val_precision = precision_score(val_targ, val_predict, average="macro")
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return
