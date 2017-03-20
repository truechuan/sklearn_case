#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from itertools import islice
import csv

# 使用逻辑回归预测订单是否成交

orderInfo = []
orderDeal = []

# 训练模型
f = open('resource/learn/orderDeal.csv', 'rU')
lines = islice(csv.reader(f), 1, None)

for row in lines:
    single = []
    single.append(int(row[5]))  # 是否指导厂商
    single.append(int(row[6]))  # 指导厂商id
    single.append(int(row[7]))  # 订单分类
    single.append(int(row[8]))  # 清洗结果
    single.append(int(row[9]))  # 用户等级
    single.append(int(row[10]))  # 团长邀约结果
    orderDeal.append(row[11])  # 成交结果
    orderInfo.append(single)

model = LogisticRegression()
model.fit(orderInfo, orderDeal)

# 测试模型准确率
f = open('resource/test/orderDeal_2yue.csv', 'rU')
lines = islice(csv.reader(f), 1, None)

_success = 0
_error = 0
for row in lines:
    single = []
    single.append(int(row[5]))  # 是否指导厂商
    single.append(int(row[6]))  # 指导厂商id
    single.append(int(row[7]))  # 订单分类
    single.append(int(row[8]))  # 清洗结果
    single.append(int(row[9]))  # 用户等级
    single.append(int(row[10]))  # 团长邀约结果

    mResult = model.predict(np.vstack(single).T)
    mResult = mResult[0]
    if mResult == row[11]:
        _success += 1
        # print '成交预测正确订单id[%s],订单成交类型[%s],机器预测类型[%s]' % (row[0], row[10], mResult)
        # print '预测正确'
    else:
        _error += 1
        print '成交预测错误订单id[%s],订单成交类型[%s],机器预测类型[%s]' % (row[0], row[11], mResult)
        # print '预测错误'

print '预测成功数据：' + str(_success)
print '预测错误数据：' + str(_error)

# print(classifier.predict([[4,9,9]]))
