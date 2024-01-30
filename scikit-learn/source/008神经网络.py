from sklearn.datasets import load_digits # 从scikit-learn中加载手写数字数据集。
from sklearn.neural_network import MLPClassifier # 多层感知机分类器，用于建立神经网络模型。
from sklearn.model_selection import train_test_split # 用于将数据集分割为训练集和测试集。
from sklearn.metrics import accuracy_score # 用于计算模型预测的准确率。


# 数据生成
data = load_digits()
X = data.images.reshape(len(data.images), -1)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = MLPClassifier(hidden_layer_sizes=(16, ))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_pred, y_test))
