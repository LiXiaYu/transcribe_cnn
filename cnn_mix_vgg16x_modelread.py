import os

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

def main():
    py_filename = "cnn_mix_vgg16x"
    path_net = os.path.join('/', 'public', 'home', 'liqi', 'data', 'analysis', 'transcribe_CNN')
    thetime = '1572186474'

    # model = keras.models.load_model(os.path.join(path_net, 'model', py_filename, thetime + r'_' + py_filename +'.model'))
    # weights_conv1D_1 = model.get_layer('conv1D_1').get_weights()

    # print(weights_conv1D_1)

    x_test_text = np.load(
        os.path.join(path_net, 'result', py_filename, thetime + r'_' + py_filename, 'x_test_text.npy'))
    x_test_numerics = np.load(
        os.path.join(path_net, 'result', py_filename, thetime + r'_' + py_filename, 'x_test_numeric.npy'))
    y_test_result = np.load(
        os.path.join(path_net, 'result', py_filename, thetime + r'_' + py_filename, 'y_test_result.npy'))
    y_test = np.load(
        os.path.join(path_net, 'result', py_filename, thetime + r'_' + py_filename, 'y_test.npy'))
    np.savetxt(os.path.join(path_net, 'result', py_filename, thetime + r'_' + py_filename, 'y_test_result'), y_test_result)
    np.savetxt(os.path.join(path_net, 'result', py_filename, thetime + r'_' + py_filename, 'y_test'), y_test)
    return 0


if __name__ == '__main__':
    main()