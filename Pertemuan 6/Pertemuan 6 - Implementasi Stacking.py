import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Bagging
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)

bagging.fit(X_train, y_train)
bagging_pred = bagging.predict(X_test)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# AdaBoost
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)

ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

# Base Models
base_models = [
    ('rf', RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )),

    ('svm', SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    )),

    ('knn', KNeighborsClassifier(
        n_neighbors=5
    )),

    ('dt', DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
]

# Meta Learner
meta_learner = LogisticRegression(max_iter=1000)

# Stacking
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba'
)

stacking.fit(X_train, y_train)

stacking_pred = stacking.predict(X_test)

# Hasil Akurasi
print("=" * 50)
print("PERBANDINGAN SEMUA METODE")
print("=" * 50)

print(f"Decision Tree     : {accuracy_score(y_test, dt_pred):.4f}")
print(f"Bagging           : {accuracy_score(y_test, bagging_pred):.4f}")
print(f"Random Forest     : {accuracy_score(y_test, rf_pred):.4f}")
print(f"AdaBoost          : {accuracy_score(y_test, ada_pred):.4f}")
print(f"Gradient Boosting : {accuracy_score(y_test, gb_pred):.4f}")
print(f"Stacking          : {accuracy_score(y_test, stacking_pred):.4f}")

# Visualisasi
models = [
    'DT',
    'Bagging',
    'RF',
    'AdaBoost',
    'GB',
    'Stacking'
]

scores = [
    accuracy_score(y_test, dt_pred),
    accuracy_score(y_test, bagging_pred),
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, ada_pred),
    accuracy_score(y_test, gb_pred),
    accuracy_score(y_test, stacking_pred)
]

plt.figure(figsize=(10, 5))

plt.bar(models, scores)

plt.ylabel('Accuracy')
plt.title('Perbandingan Akurasi')

plt.ylim(0.9, 1.0)

for i, v in enumerate(scores):
    plt.text(i, v + 0.002, f'{v:.4f}', ha='center')

plt.show()