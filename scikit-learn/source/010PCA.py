from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


# Load the Iris dataset
data_iris = load_iris()

# PCA assumes that the dataset is centered, subtract the mean of each variable
data_iris_centered = data_iris.data - np.mean(data_iris.data, axis=0)

# Define PCA model and fit to the centered iris dataset (A)
pca_A = PCA(n_components=len(data_iris_centered[0]))
pca_A.fit(data_iris_centered)
cumulative_contributions_A = np.cumsum(pca_A.explained_variance_ratio_)

# For dataset B, we are not applying PCA, just using the variance of the original dataset
# Since PCA is not applied, the "contribution" of each original variable is just its variance ratio
variance_B = np.var(data_iris_centered, axis=0)
total_variance_B = np.sum(variance_B)
contribution_B = variance_B / total_variance_B
cumulative_contributions_B = np.cumsum(contribution_B)

# Plotting the cumulative contribution rates for A and B
plt.figure(figsize=(12, 7))
plt.plot(range(1, len(cumulative_contributions_A) + 1), cumulative_contributions_A, '-o', label='A（应用PCA的累计贡献率）')
plt.plot(range(1, len(cumulative_contributions_B) + 1), cumulative_contributions_B, '-o', label='B（未应用PCA的方差比例）')

plt.title('累计贡献率对比')
plt.xlabel('主成分数')
plt.ylabel('累计贡献率')
plt.legend()
plt.grid(True)
plt.show()
