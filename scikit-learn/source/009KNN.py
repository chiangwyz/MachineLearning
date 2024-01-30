import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# 数据生成
# X, y = datasets.make_moons(noise=0.3, random_state=0)
X, y = datasets.make_moons(noise=0.3)


# 拆分训练和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# KNN模型训练
k = 5  # 假定最近邻的数量为5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)
# 评估
accuracy_score(y_pred, y_test)

# 创建彩色地图
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

# 绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# 将结果放入彩色图
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# 绘制训练点
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"2-Class classification (k = {k}, weights = 'uniform')")

plt.show()