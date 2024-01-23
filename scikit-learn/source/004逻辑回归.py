"""
逻辑回归，代码解释见对应文档
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


X_train = np.r_[np.random.normal(3, 1, size=50), np.random.normal(-1, 1, size=50)].reshape((100, -1))
y_train = np.r_[np.ones(50), np.zeros(50)]

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.predict_proba([[-1], [0], [1], [2], [3]])[:])
