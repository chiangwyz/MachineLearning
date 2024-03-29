# K-means算法

把相似的数据汇总为簇的方法叫做聚类。

K-means算法是一种聚类算法，被广泛应用于数据分析。

## 概述

K-means算法是一种有代表性的聚类算法，常用于比较大的数据集，所以在市场分析和计算机视觉等领域得到了广泛的应用。

通过计算数据点与各重心的距离，找出离得最近的醋的重心，可以确定数据点所属的簇，求簇的重心是K-means算法中重要的计算。

## 算法说明

K-means算法的典型计算步骤如下：

1. 从数据点中随机选择数量与簇的数量相同的数据点，作为这些簇的重心。

2. 计算数据点与各重心之间的距离，并将最近的重心所在簇作为该数据点所属的簇。

3. 计算每个簇的数据点的平均值，并将其作为新的重心。

4. 重复步骤2和步骤3，继续计算，直到所有数据点不改变所属的簇，或者达到最大计算步数。

步骤1中的簇的数量是一个超参数，需要再训练时设置，对于有些数据集，我们可能很难决定要设置的簇的数量，在这种情况下，可以使用Elbow方法(肘方法)。

另外，有时候选择的重心不好可能会导致步骤2和步骤3的训练无法顺利进行。比如在随机选择重心之间太近等情况下，就会出现这种情况，使用K-means++等方法，选择位置尽可能远离的数据点作为重心的初始值，可以解决这个问题。

### 示例代码

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris


data = load_iris()
n_clusters = 3 # 将簇的数量设置为3
model = KMeans(n_clusters=n_clusters)
model.fit(data.data)
print(model.labels_) # 各数据点所属的簇 
print(model.cluster_centers_) # 通过fit()计算得到的重心
```


## 算法说明

### 聚类结果的评估方法

如何判断聚类结果是好是坏。

簇内结果的好坏可以通过计算簇内平方和(With-in Cluster of Squares, WCSS)来定量评估(这个WCSS随着簇的数量增加而变小，所以可以用于相同数量的簇的情况下的比较)。

WCSS指的的是对所有簇计算其所属的数据点与簇的重心之间距离的平方和，并将他们相加得到的值，这个值越少，说明聚类结果越好。


### 使用Elbow方法确定簇的数量

在使用K-means算法时，首先要确定的就是簇的数量这个超参数，但有时我们并不知道应该将簇的值设定为多少合适。确定合理的簇的数量的一个方法就是Elbow方法，我们知道，随着簇的数量增加，WCSS会变小，但有时WCSS的变小幅度会从簇的数量为某个值时开始放缓。

比如在簇的数量增加到3的过程中，WCSS明显变小，但当簇的数量逐渐增加为4,5,... 时，可以看到WCSS变小幅度变缓了，Elbow方法将图形中看起来像手臂弯曲时的肘部的点作为簇的数量。

当没有很明确的理由来确定簇的数量或对数据知之甚少时，Elbow方法很有用，不过在实际的分析中，尝尝不会出现图中那样明显的“肘部”，所以Elbow方法的结果不过是一种参考而已。