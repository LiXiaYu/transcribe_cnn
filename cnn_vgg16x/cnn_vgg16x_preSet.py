import typing
import os
import numpy as np

from cnn_vgg16x import cnn_vgg16x


class cnn_vgg16x_preSet(cnn_vgg16x.cnn_vgg16x):
    def __init__(self):
        super().__init__()
        self.py_filename = "cnn_vgg16x_preSet"
        self.CreateDir()

    def PreSet(self):
        data = self.xys_c_end
        iop_r = data['Right_Intraocular_Pressure']
        iop_l = data['Left_Intraocular_Pressure']

        icp = 0.44 * data['Width'] / data['Height'] / data['Height'] + 0.16 * data['Blood_Pressure_Diastolic'] - 0.18 * data['Years'] - 1.91

        tlcpd_r = iop_r - icp
        tlcpd_l = iop_l - icp
        data.insert(data.shape[1] - 1, 'Left_TLCPD', tlcpd_l)
        data.insert(data.shape[1] - 1, 'Right_TLCPD', tlcpd_r)
        data = data.replace([np.inf, -np.inf], np.nan)
        self.xys_c_end = data.fillna(data.mean())
        # data['TLCPD'] = tlcpd_re
        return

    def SplitData(self, index_load: str = "false"):
        self.PreSet()
        return super().SplitData(index_load)


def main():
    ca = cnn_vgg16x_preSet()

    # ca.Fixeddata_Fix()

    ca.Run(index_load="true")
    return 0


if __name__ == '__main__':
    main()
