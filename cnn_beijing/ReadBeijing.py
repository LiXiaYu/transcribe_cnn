import typing
import time, datetime

import numpy as np
import pandas as pd

import pymysql

import KNNcommon as Kc


def ReadBeijingFromMySqlToDataFrame():
    conn = pymysql.connect(host='201-files-vpn.ccmubme.ccmu.edu.cn', port=3306, user='liqi', passwd='66554321',
                           db='beijing', charset='utf8')
    cursor = conn.cursor()

    cursor.execute(
        "select * from ge1 a left join ge2 b on a.aPID=b.bPID left join ge3 c on a.aPID=c.cPID left join ge4 d on a.aPID=d.dPID")
    data = cursor.fetchall()

    return data


def ReadBeijingFromMySqlToXY(nameXs: typing.List[str], nameY: str) -> pd.DataFrame:
    conn = pymysql.connect(host='201-files-vpn.ccmubme.ccmu.edu.cn', port=3306, user='liqi', passwd='66554321',
                           db='beijing', charset='utf8')
    cursor = conn.cursor()

    sql_str = "select " + nameY + ","
    for i in range(len(nameXs) - 1):
        sql_str += nameXs[i] + ","
    sql_str += nameXs[len(nameXs) - 1]
    sql_str += " from ge1 a left join ge2 b on a.aPID=b.bPID left join ge3 c on a.aPID=c.cPID left join ge4 d on " \
               "a.aPID=d.dPID "

    cursor.execute(sql_str)
    yx_values = cursor.fetchall()

    nameXs.insert(0, nameY)
    data = pd.DataFrame(yx_values)
    data.columns = nameXs
    return data


def main():
    # data = ReadBeijingFromMySqlToXY(
    #     ["f_GENDER", "f_BIRTH", "f_BPPUL1", "f_BPSYS1", "f_BPDIA1", "f_WTKG", "f_HTCM", "f_IOPNCTR1", "f_IOPNCTL1"],
    #     "f_HISGLAUC")

    data = pd.read_csv("data.csv")

    data = data[~data['f_GENDER'].isin([None])]

    dlis = []
    for temp_index_i in range(1, data.shape[1] - 3):
        if int(data.iat[data.shape[0] - 1, temp_index_i]) < ((data.shape[0] - 3) * 2 / 3):
            dlis.append(temp_index_i)
    data = data.drop(data.columns[dlis], axis=1)
    dlis = []
    for temp_index_i in range(data.shape[0] - 3):
        if int(data.iat[temp_index_i, data.shape[1] - 1]) < ((data.shape[1] - 3) * 2 / 3):
            dlis.append(temp_index_i)
    data = data.drop(dlis, axis=0)
    data = data.reset_index()
    data = data.drop(data.columns[[data.shape[1] - 3, data.shape[1] - 2, data.shape[1] - 1]], axis=1)
    data = data.drop([data.shape[0] - 3, data.shape[0] - 2, data.shape[0] - 1], axis=0)

    l = []
    for a, b in zip(data['f_main_Date'], data['f_BIRTH']):
        try:
            y = time.strptime(a, '%d-%b-%y').tm_year - time.strptime(b, '%d-%b-%y').tm_year
            if y < 0:
                y = 100 + y
            l.append(y)
        except:
            l.append(np.nan)

    data['year'] = l
    data.dropna(axis=0, how='all')
    data.dropna(axis=1, how='all')

    data = data.drop(data.columns[[0, 1]], axis=1)
    data = data.replace("#NULL!", np.nan)

    l_n = []
    l_t = []
    for index, row in data.iteritems():
        try:
            data[index] = row.astype('float')
            l_n.append(index)

            try:
                # 顺便填补均值
                group = data[index]
                values = np.nanmean(group.values)
                group.fillna(value=values, inplace=True)  # 参看2，填充空值，value这里选择为字典形式，字典的key指明列，字典的value指明填充该列所用的值
                data[index] = group
            except :
                aaaa=11
        except:
            data[index] = row.astype('str')
            l_t.append(index)

    # mice暂时出错
    # data_n = Kc.Interpolation_mice(data[l_n])
    # 先用均值填
    # 在上面
    data_n = data.drop(l_t, axis=1)
    data_t = data.drop(l_n, axis=1)
    xys_mi = pd.concat([data_t, data_n], axis=1)

    # 数据插补
    print("开始插补数据")
    # #多重插补
    # xys_mi = Kc.Interpolation_mice(data)
    print("插补数据完成")
    ############

    xys_mi.to_csv("Beijing_dataAll_nanmean.csv")
    return 0


if __name__ == '__main__':
    main()
