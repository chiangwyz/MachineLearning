from sklearn.naive_bayes import MultinomialNB

# 数据生成
X_train = \
[[1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1]]
y_train = [1, 1, 1, 0, 0, 0]
model = MultinomialNB()
model.fit(X_train, y_train) # 训练
result = model.predict([[1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]) # 评估

print("result =", result)

