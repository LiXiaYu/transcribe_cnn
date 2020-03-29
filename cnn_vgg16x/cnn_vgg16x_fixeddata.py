import typing
import os

import numpy as np
import pandas as pd
import gensim
from tensorflow import keras

import transcribe_common as Kc

from cnn_vgg16x import cnn_vgg16x


class cnn_vgg16x_fixeddata(cnn_vgg16x.cnn_vgg16x):
    def __init__(self):
        super().__init__()
        self.py_filename = "cnn_vgg16x_fixeddata"
        self.CreateDir()

    def Run(self):

        self.Fixeddata_Load()

        self.TrainNet()

        self.PostprocessData()

        return

    def Fixeddata_Fix(self):

        self.PreprocessData_Load()

        self.SplitData()

        self.SeparateData()

        self.Fixeddata_Save()
        return

    def Fixeddata_Save(self):
        np.save(os.path.join(self.path_net, 'x_train_text.npy'), self.x_train_text)
        np.save(os.path.join(self.path_net, 'x_train_numeric.npy'), self.x_train_numeric)

        np.save(os.path.join(self.path_net, 'x_test_text.npy'), self.x_test_text)
        np.save(os.path.join(self.path_net, 'x_test_numeric.npy'), self.x_test_numeric)

        np.save(os.path.join(self.path_net, 'y_train_all.npy'), self.y_train_all)
        np.save(os.path.join(self.path_net, 'y_test_all.npy'), self.y_test_all)

        return

    def Fixeddata_Load(self):
        self.x_train_text = np.load(os.path.join(self.path_net, 'x_train_text.npy'))
        self.x_train_numeric = np.load(os.path.join(self.path_net, 'x_train_numeric.npy'))

        self.x_test_text = np.load(os.path.join(self.path_net, 'x_test_text.npy'))
        self.x_test_numeric = np.load(os.path.join(self.path_net, 'x_test_numeric.npy'))

        self.y_train_all = np.load(os.path.join(self.path_net, 'y_train_all.npy'))
        self.y_test_all = np.load(os.path.join(self.path_net, 'y_test_all.npy'))

        return

def main():
    ca = cnn_vgg16x_fixeddata()

    ca.Fixeddata_Fix()

    ca.Run()
    return 0

if __name__ == '__main__':
    main()
