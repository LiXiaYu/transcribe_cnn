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


class cnn_common(object):
    def __init__(self):

        self.SetExternalVariable(str(int(time.time())),
                                 "cnn_common",
                                 os.path.join('/', 'public', 'home', 'liqi', 'data', 'analysis', 'transcribe_CNN'))


    def SetExternalVariable(self, _nowtime: str, _py_filename: str, _path_net: str):
        self.nowtime = _nowtime
        self.py_filename = _py_filename
        self.path_net = _path_net

    def PreprocessData(self):

        return

    def SplitData(self):

        return

    def CreateDir(self):
        Kc.CreateDir(os.path.join(self.path_net, 'result', self.py_filename))
        Kc.CreateDir(os.path.join(self.path_net, 'model', self.py_filename))

        return

    def Run(self):

        return
