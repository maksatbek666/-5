# -5
# K-Nearest Neighbors (KNN) Классификатор для Датасета Iris

Этот проект демонстрирует использование классификатора K-Nearest Neighbors (KNN) для оценки датасета Iris с помощью кросс-валидации. Точность классификатора оценивается для различных значений `k` (число соседей) в диапазоне от 70 до 120.

## Файлы

### main.py
Основной скрипт, содержащий следующие шаги:
1. Загрузка датасета Iris.
2. Инициализация классификатора KNN.
3. Проведение кросс-валидации для различных значений `k`.
4. Вывод точности для каждого значения `k`.

python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier

# Загрузка датасета Iris
iris = load_iris()
X, y = iris.data, iris.target

# Инициализация классификатора KNN
knn = KNeighborsClassifier()

# Определение диапазона значений k для оценки
k_values = list(range(70, 120))
scores = []

# Оценка KNN для каждого значения k с помощью кросс-валидации
for k in k_values:
    knn.set_params(n_neighbors=k)
    cv_results = cross_validate(knn, X, y, cv=5, scoring='accuracy', return_train_score=True)
    test_score_mean = cv_results['test_score'].mean()
    scores.append(test_score_mean)

# Вывод результатов
for k, score in zip(k_values, scores):
    print(f"k = {k}, Точность: {score:.4f}")


## Требования

Для запуска этого проекта вам потребуется установить Python и следующие библиотеки:
- scikit-learn




## Результаты

Скрипт выведет точность классификатора KNN для каждого значения `k` в указанном диапазоне. Это можно использовать для определения оптимального числа соседей для классификатора KNN на датасете Iris.
