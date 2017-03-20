#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import binascii
from itertools import islice
import csv


# 使用knn预测订单分类


def string2int(s):
    return int(binascii.b2a_hex('陈川'), 16)


orderInfo = []
orderClass = []

# 训练模型
f = open('resource/learn/orderClass.csv', 'rU')
lines = islice(csv.reader(f), 1, None)

for row in lines:
    single = []
    single.append(int(row[4]))  # 是否活动当天到店
    single.append(int(row[5]))  # 订单来源
    single.append(int(row[6]))  # 是否到店
    single.append(int(row[7]))  # 是否成交
    single.append(int(row[8]))  # 是否本地用户
    orderClass.append(row[10])  # 订单类型
    orderInfo.append(single)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(orderInfo, orderClass)

# 测试模型准确率
f = open('resource/test/orderClass_2yue.csv', 'rU')
lines = islice(csv.reader(f), 1, None)

_success = 0
_error = 0
for row in lines:
    single = []
    single.append(int(row[4]))  # 是否活动当天到店
    single.append(int(row[5]))  # 订单来源
    single.append(int(row[6]))  # 是否到店
    single.append(int(row[7]))  # 是否成交
    single.append(int(row[8]))  # 是否本地用户

    mResult = model.predict(np.vstack(single).T)
    mResult = mResult[0]
    if mResult == row[10]:  # 订单类型
        _success += 1
        # print '订单类型预测正确订单id[%s],订单真实类型[%s],机器预测类型[%s]' % (row[0], row[10], mResult)
        # print '预测正确'
    else:
        _error += 1
        print '订单类型预测错误订单id[%s],订单真实类型[%s],机器预测类型[%s]' % (row[0], row[10], mResult)

        # print '预测错误'
print '预测成功数据：' + str(_success)
print '预测错误数据：' + str(_error)






# print(neigh.predict([[4,9,9]]))
# print(neigh.predict_proba([[2.5,4,6]]))
