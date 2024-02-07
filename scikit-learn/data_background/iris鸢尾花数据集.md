load_iris() 是 Scikit-learn 库中的一个函数，用于加载著名的鸢尾花（Iris）数据集。这个数据集是模式识别最著名的数据集之一，最初由罗纳德·费舍尔在 1936 年发布。

数据集包含 150 个样本，每个样本都是关于一朵鸢尾花的测量数据，包括：
1. 花萼长度（sepal length）
2. 花萼宽度（sepal width）
3. 花瓣长度（petal length）
4. 花瓣宽度（petal width）

这些测量值都是以厘米为单位。数据集中的花被分为三个类别，每个类别50个样本：

1. Setosa（山鸢尾）
2. Versicolor（变色鸢尾）
3. Virginica（维吉尼亚鸢尾）

load_iris() 函数返回的数据是一个类似字典的对象，包含以下关键组件：
1. data: 数据集的特征数组，维度为 [n_samples, n_features]，即 150x4。
2. target: 每朵花对应的类别（0, 1, 或 2），维度为 [n_samples]。
3. target_names: 类别的名称，即 ['setosa', 'versicolor', 'virginica']。
4. feature_names: 特征的名称列表。
5. DESCR: 数据集的全面描述。
6. filename: 数据集文件的路径（如果有）。

这个数据集经常被用作机器学习和统计分类技术的入门案例研究。