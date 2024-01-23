import numpy as np

# 从sklearn.preprocessing导入PolynomialFeatures，用于生成多项式特征
from sklearn.preprocessing import PolynomialFeatures
# 从sklearn.linear_model导入Ridge，用于进行岭回归分析
from sklearn.linear_model import Ridge
# 从sklearn.metrics导入mean_squared_error，用于计算均方误差
from sklearn.metrics import mean_squared_error

train_size = 200
test_size = 120

train_X = np.random.uniform(low=0, high=1.2, size=train_size)
test_X = np.random.uniform(low=0, high=1.2, size=test_size)
train_y = np.sin(train_X * 2 * np.pi) + np.random.normal(0, 0.2, train_size)
test_y = np.sin(test_X * 2 * np.pi) + np.random.normal(0, 0.2, test_size)

poly = PolynomialFeatures(6)  # 次数为6
train_poly_X = poly.fit_transform(train_X.reshape(train_size, 1))
test_poly_X = poly.fit_transform(test_X.reshape(test_size, 1))

model = Ridge(alpha=1.0)
model.fit(train_poly_X, train_y)
train_pred_y = model.predict(train_poly_X)
test_pred_y = model.predict(test_poly_X)
print(mean_squared_error(train_pred_y, train_y))
print(mean_squared_error(test_pred_y, test_y))
