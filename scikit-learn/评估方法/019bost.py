import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])

# 只使用13个特征变量中的"住宅平均房间数(列名为RM)"
data = data[:, 5:6]
target = raw_df.values[1::2, 2]

np.set_printoptions(precision=5, suppress=True)


print("data", data[:2, :])
print("target", target)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(data, target)
y_pred = model.predict(data)

print("model.coef =", model.coef_)
print("model.intercept_ =", model.intercept_)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(data, target, color='pink', marker='s', label='data set')
ax.plot(data, y_pred, color='blue', label='LinearRegression')
ax.legend()
plt.show()
# plt.savefig("boston house rent price prediction.pdf")

from sklearn.metrics import mean_squared_error
mean_sqaured = mean_squared_error(target, y_pred)

print("mean sqaure error =", mean_sqaured)


from sklearn.svm import SVR

model_svr = SVR(C=0.01, kernel='linear')
model_svr.fit(data, target)
y_svr_pred = model_svr.predict(data)



import matplotlib.pyplot as plt
fig1, ax1 = plt.subplots()
ax1.scatter(data, target, color='pink', marker='s', label='data set')
ax1.plot(data, y_pred, color='blue', label='LinearRegression')
ax1.plot(data, y_svr_pred, color='red', label='SVR')
ax1.legend()
plt.show()