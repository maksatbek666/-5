
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X, y = iris.data, iris.target

knn = KNeighborsClassifier()
k_values = list(range(70, 120))
scores = []

for k in k_values:
    knn.set_params(n_neighbors=k)
    cv_results = cross_validate(knn, X, y, cv=5, scoring='accuracy', return_train_score=True)
    test_score_mean = cv_results['test_score'].mean()
    scores.append(test_score_mean)

for k, score in zip(k_values, scores):
    print(f"k = {k}, Accuracy: {score:.4f}")



