import tensorflow as tf
import numpy as np
import pandas as pd

import pymysql


def ReadFromMySqlToDataFrame():
    writer = tf.io.TFRecordWriter('test.tfrecords')

    conn = pymysql.connect(host='201-files-vpn.ccmubme.ccmu.edu.cn', port=3306, user='liqi', passwd='66554321',
                           db='transcribe_1', charset='utf8')
    cursor = conn.cursor()

    cursor.execute(
        "select COLUMN_NAME from information_schema.COLUMNS where table_name = 'origindata_select' and table_schema='transcribe_1'")
    keyNames = cursor.fetchall()
    keyNames = list(map(lambda a: str(a[0]), keyNames))

    def temp_lambda_i32juiweojfiosdfjsdioajfkwpe(a):
        cursor.execute("select " + a + " from transcribe_1.origindata_select")
        return [','.join(map(str, tp)) for tp in list(cursor.fetchall())]

    datas = list(map(temp_lambda_i32juiweojfiosdfjsdioajfkwpe, keyNames))

    datas = list(map(list, zip(*datas)))

    dataFrame = pd.DataFrame(datas, columns=keyNames)
    return dataFrame


def To_Tfrecord():
    writer = tf.io.TFRecordWriter('test.tfrecords')

    conn = pymysql.connect(host='201-files.ccmubme.ccmu.edu.cn', port=3306, user='liqi', passwd='66554321',
                           db='transcribe_1', charset='utf8')
    cursor = conn.cursor()

    cursor.execute(
        "select COLUMN_NAME from information_schema.COLUMNS where table_name = 'origindata_image_1_7' and table_schema='transcribe_1'")
    keyNames = cursor.fetchall()
    keynames = list(map(lambda a: a[0], keyNames))
    cursor.execute("select * from transcribe_1.origindata_image_1_7")
    result1 = cursor.fetchall()
    for i in range(len(result1)):
        data = result1[i]

        # writer.write(data)
    # writer.close()

    cursor.close()
    conn.close()

    return keynames, result1


def ReadFromMySqlToDataFrame(database: str, tablename: str) -> pd.DataFrame:
    conn = pymysql.connect(host='201-files-vpn.ccmubme.ccmu.edu.cn', port=3306, user='liqi', passwd='66554321',
                           db=database, charset='utf8')
    cursor = conn.cursor()
    cursor.execute(
        "select COLUMN_NAME from information_schema.COLUMNS where table_name ='" + tablename + "' and table_schema ='" + database + "';")

    keyNames = cursor.fetchall()
    keyNames = list(map(lambda a: str(a[0]), keyNames))

    def temp_lambda_i32juiweojfiosdfjsdioajfkwpe(a):
        cursor.execute("select " + a + " from " + database + "." + tablename + ";")
        return [','.join(map(str, tp)) for tp in list(cursor.fetchall())]

    datas = list(map(temp_lambda_i32juiweojfiosdfjsdioajfkwpe, keyNames))

    datas = list(map(list, zip(*datas)))

    dataFrame = pd.DataFrame(datas, columns=keyNames)

    return dataFrame

def main():
    To_Tfrecord()


if __name__ == '__main__':
    main()
