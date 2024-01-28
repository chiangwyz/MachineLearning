from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np


# 数据生成
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_pred, y_test)


# 获取特征重要性并排序
indices = np.argsort(model.feature_importances_)
sorted_importances = model.feature_importances_[indices]
sorted_features = [data.feature_names[i] for i in indices]

print("indices type =", indices.dtype)
print("indices =\n", indices)
print("sorted_importances = \n", sorted_importances)
print("sorted_features = \n", sorted_features)


# 创建条形图
plt.figure(figsize=(20, 12))
plt.barh(sorted_features, sorted_importances)
plt.xlabel('Feature Importance', fontsize=4)
plt.ylabel('Feature', fontsize=4)
plt.title('Feature Importance in Wine Dataset')
plt.show()
