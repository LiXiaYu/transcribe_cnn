from cnn_vgg16x import cnn_vgg16x


class cnn_vgg16x_noiop(cnn_vgg16x.cnn_vgg16x):
    def __init__(self):
        super().__init__()
        self.py_filename = "cnn_vgg16x_preSet"
        self.CreateDir()

    def PreSet(self):
        data = self.xys_c_end

        data.drop('Right_Intraocular_Pressure', axis=1)
        data.drop('Left_Intraocular_Pressure', axis=1)

        # data['TLCPD'] = tlcpd_re
        return

    def SplitData(self, index_load: str = "false"):
        self.PreSet()
        return super().SplitData(index_load)


def main():
    ca = cnn_vgg16x_noiop()

    # ca.Fixeddata_Fix()

    ca.Run(index_load="true")
    return 0


if __name__ == '__main__':
    main()
