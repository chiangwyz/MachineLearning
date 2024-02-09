from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

data = load_iris()

n_components = 3

model = GaussianMixture(n_components=n_components)

model.fit(data.data)


print(model.predict(data.data))
print(model.means_)
print(model.covariances_)