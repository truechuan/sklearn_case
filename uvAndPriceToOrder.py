#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import islice
import csv

# 使用多元线性预测流量和订单的关系

uvCount = []
orderCount = []
calOrderCount = []
price = []

f = open('resource/learn/activityInfo.csv', 'rb')
lines = islice(csv.reader(f), 1, None)
for row in lines:
    uvCount.append(int(row[1]))
    orderCount.append(int(row[2]))
    price.append(int(row[3]))
uvCount = np.array(uvCount)
orderCount = np.array(orderCount)
price = np.array(price)
# print uvCount
# print orderCount
# print price

# uv与订单关系
model = LinearRegression()
x = np.vstack([uvCount, price]).T
model.fit(x, orderCount)

inputUv = int(raw_input("请输入UV: "))
inputPrice = int(raw_input("请输入价格: "))

condition = np.vstack([inputUv, inputPrice]).T

print '订单量：' + str(model.predict(condition))


# fig = plt.figure()
# ax = Axes3D(fig)
# X, Y, Z = uvCount, price, orderCount
#
# ax.scatter(X, Y, Z, c='r')
#
# for index in range(len(uvCount)):
#     print index
#     calOrderCount.append(model.predict([uvCount[index], price[index]]))
# calOrderCount = np.array(calOrderCount)
#
# ax.scatter(X, Y, Z, c='r')
# ax.plot_surface(X, Y, calOrderCount, rstride=1, cstride=1, cmap='rainbow')
#
# plt.show()
