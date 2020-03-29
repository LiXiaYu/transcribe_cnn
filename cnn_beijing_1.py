import typing
import time
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf

import transcribe_common as Kc

def main():
    path_net = os.path.join('/', 'public', 'home', 'liqi', 'data', 'analysis', 'transcribe_CNN')
    # path_net = os.path.join('Z:\\', 'datas', 'analysis')

    # dataRW()
    ge_all = pd.read_csv(os.path.join(path_net, r'beijing_ge_all.csv'),low_memory=False)
    ge_all.drop(ge_all.columns[[0]], axis=1, inplace=True)  # 删除多出来的序号列

    return 0

def dataRW():
    # path_net = os.path.join('/', 'public', 'home', 'liqi', 'data', 'analysis', 'transcribe_CNN')
    path_net = os.path.join('Z:\\', 'datas', 'analysis')

    from ReadDatas import ReadFromMySqlToDataFrame
    ge = []
    ge.append(ReadFromMySqlToDataFrame("beijing", "ge1"))
    ge.append(ReadFromMySqlToDataFrame("beijing", "ge2"))
    ge.append(ReadFromMySqlToDataFrame("beijing", "ge3"))
    ge.append(ReadFromMySqlToDataFrame("beijing", "ge4"))

    ge_all = pd.merge(pd.merge(pd.merge(ge[0], ge[1]), ge[2]), ge[3])
    ge_all.to_csv(os.path.join(path_net, r'beijing_ge_all.csv'))

    return 0


if __name__ == '__main__':
    main()
