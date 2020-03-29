import typing
import os

import numpy as np
import random
import pandas as pd
import thulac
import gensim
import tensorflow as tf

from statsmodels.imputation import mice
import statsmodels.api as sm
from tensorflow import keras

from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_curve, roc_auc_score


# 创建文件夹
def CreateDir(flodername: str):
    isExists = os.path.exists(flodername)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(flodername)


# 根据X和Y的名字取出数据来
def DataFrameToXY(dataFrame: pd.DataFrame, nameXs: object, nameY: object, manualFilePath: str = "") -> pd.DataFrame:
    y_str = dataFrame[nameY]
    mdtoic, y_data = MDCodetoIntCode(y_str.tolist(), manualFilePath)

    # 转置list，提取x
    x_data = list(map(list, zip(*list(map(lambda a: list(dataFrame[a]), nameXs)))))

    xys = np.hstack((np.array(y_data).reshape((len(y_data), 1)), np.array(x_data)))

    datas = pd.DataFrame(xys, columns=[nameY] + nameXs)

    return datas


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
            try:
                ic.append(x.values[0])
            except IndexError:
                ic.append(-1)

        return Cs, ic


# 数值型数据插补
def Mice_numeric(xys: pd.DataFrame) -> pd.DataFrame:
    xys[list(xys.columns)] = xys[list(xys.columns)].apply(pd.to_numeric, errors='coerce')
    xys = xys.astype(float)
    # 数据插补

    # #多重插补
    xys_mi = Interpolation_mice(xys)
    return xys_mi


# 数据插补
def Interpolation_mice(df: pd.DataFrame) -> pd.DataFrame:
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


# 划分测试集和训练集
def SplitGroup(xys: np.array, test_r: int = 0.3, test_index_list: typing.List[int] = [], train_index_list: typing.List[int] = []) -> typing.Tuple[np.array, np.array, np.array, np.array]:
    # 分训练组和测试组
    index = np.arange(len(xys))
    np.random.shuffle(index)

    test_index = index[0: int(len(xys) * test_r)]
    train_index = index[int(len(xys) * test_r):len(xys)]

    # 返回分组的索引
    train_index_list.clear()
    train_index_list.append(train_index)
    test_index_list.clear()
    test_index_list.append(test_index)

    y_test = xys[test_index, 0]
    x_test = xys[test_index, 1:]

    y_train = xys[train_index, 0]
    x_train = xys[train_index, 1:]

    return x_train, y_train, x_test, y_test


def TextToVector_doc2vec(text_xys: typing.List[str]) -> typing.Tuple[
    gensim.models.Doc2Vec, typing.List[str]]:
    text_x_d2v_train = []
    thu1 = thulac.thulac(seg_only=True)  # 默认模式
    for i in range(len(text_xys)):
        try:
            index_text_y = text_xys[i]
        except KeyError:
            index_text_y = ""

        # 分词
        temp_thu1_rt = thu1.cut(index_text_y, text=True)
        document = gensim.models.doc2vec.TaggedDocument(temp_thu1_rt, tags=[i])
        text_x_d2v_train.append(document)

    model = gensim.models.Doc2Vec(text_x_d2v_train, size=50, window=8, min_count=5, workers=4)
    text_x_vds = list(map(lambda x: np.array(x), model.docvecs.vectors_docs.tolist()))

    return model, text_x_vds


# f1,recalls,precisions计算用
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


class Mertics_f1s_recalls_precisions(keras.callbacks.Callback):
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


class Mertics_roc_auc(keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_train_begin(self, logs={}):
        self.val_fprs = []
        self.val_tprs = []
        self.val_roc_aucs = []

    def on_train_end(self, logs={}):
        val_predict_onehot = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_predict = np.array(list(map(np.argmax, val_predict_onehot)))  # onehot转正常
        val_targ = self.validation_data[1]
        _fpr, _tpr, thresholds = roc_curve(val_targ, val_predict, pos_label=0)
        self.val_roc_aucs.append(thresholds)
        return
