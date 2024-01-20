from sklearn.linear_model import LinearRegression

"""
X 代表的是线性回归模型的输入数据。在机器学习和统计建模中，输入数据通常表示为一个二维数组或矩阵，其中每一行代表一个数据点，
每一列代表一个特征。
"""
X = [[10.0], [8.0], [13.0], [9.0], [11.0], [14.0], [6.0], [4.0], [12.0], [7.0], [5.0]]
y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
model = LinearRegression()
model.fit(X, y)

# 截距
print(model.intercept_)

# 斜率
print(model.coef_)

y_pred = model.predict([[0], [1]])

# 对x=0, x=1的预测结果
print(y_pred)