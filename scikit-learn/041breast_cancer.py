from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

# 加载数据
data = load_breast_cancer()
X = data.data
y = 1 - data.target
# 反转标签的0和1

# 查看数据集的描述
# 设置numpy打印选项，限制小数点后三位
np.set_printoptions(precision=3, suppress=True)
print(data.DESCR)
# print('%.2f', X[:5])

# 选择前10个特征
# X = X[:, :]

# 划分训练数据和测试数据
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化逻辑回归模型
model = LogisticRegression(max_iter=10000)

# 训练模型
model.fit(X, y)

# 在测试集上进行预测
y_pred = model.predict(X)

# 如果需要，可以评估模型性能
# 例如，使用accuracy_score来计算准确率
accuracy = accuracy_score(y, y_pred)
print('Accuracy: {}'.format(accuracy))

from sklearn.metrics import precision_score

precision = precision_score(y, y_pred)
print('precision: {}'.format(precision))

from sklearn.metrics import recall_score
recall = recall_score(y, y_pred)
print('recall: {}'.format(recall))

from sklearn.metrics import f1_score
f1 = f1_score(y, y_pred)
print('f1: {}'.format(f1))

proba = model.predict_proba(X)
print("proba = ", proba)

y_pred2 = (model.predict_proba(X)[:, 1]>0.1).astype(np.int)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, y_pred)
print(cm)
