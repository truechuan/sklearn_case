#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from itertools import islice
import csv

# 使用一元线性预测流量和订单的关系

uvCount = []
orderCount = []
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
print uvCount
print orderCount
print price

# uv与订单关系
model = LinearRegression()
x = np.vstack([uvCount]).T
model.fit(x, orderCount)

plt.figure()
## plt.subplot(221)
plt.title('uv and order')
plt.scatter(x, orderCount, color='red')
plt.plot(x, model.predict(x), color='blue')

plt.xticks(())
plt.yticks(())
plt.axis('tight')
plt.xlabel('uv')
plt.ylabel('order')

plt.show()
