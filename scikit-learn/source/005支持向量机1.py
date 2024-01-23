from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

# 数据生成
centers = [(-1, -0.125), (0.5, 0.5)]
X, y = make_blobs(n_samples=50, n_features=2, centers=centers, cluster_std=0.3)
print("X =\n", X)
print("y =\n", y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = LinearSVC()
model.fit(X_train, y_train) # 训练
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test) # 评估
print("accuracy =", accuracy)

# 可视化
# 计算决策边界直线
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(x_min, x_max)
yy = a * xx - (model.intercept_[0]) / w[1]

# 绘制决策边界直线和数据点
plt.figure(figsize=(8, 6))
plt.plot(xx, yy, 'k-', label="Decision Boundary")
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, edgecolor='k', cmap='viridis', label="Training Data")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, edgecolor='k', cmap='viridis', alpha=0.6, label="Test Data")
plt.title("SVM Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()