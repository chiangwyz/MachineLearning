import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_20newsgroups


categories = ['misc.forsale', 'rec.autos','comp.graphics', 'sci.med']
remove = ('headers', 'footers', 'quotes')
twenty_train = fetch_20newsgroups(subset='train',
                                  remove=remove,
                                  categories=categories) # 训练数据
twenty_test = fetch_20newsgroups(subset='test',
                                 remove=remove,
                                 categories=categories) # 验证数据


count_vect = CountVectorizer() # 单词出现次数
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_test_count = count_vect.transform(twenty_test.data)

model = LinearSVC(max_iter=10000)
model.fit(X_train_counts, twenty_train.target)
predicted = model.predict(X_test_count)
value = np.mean(predicted == twenty_test.target)
print("value =", value)


tf_vec = TfidfVectorizer()  # tf-idf
X_train_tfidf = tf_vec.fit_transform(twenty_train.data)
X_test_tfidf = tf_vec.transform(twenty_test.data)

model = LinearSVC()
model.fit(X_train_tfidf, twenty_train.target)
predicted = model.predict(X_test_tfidf)
value2 = np.mean(predicted == twenty_test.target)
print("value2 =", value2)
