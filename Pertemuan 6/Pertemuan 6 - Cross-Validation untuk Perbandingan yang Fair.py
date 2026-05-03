import time
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, KFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
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

# Base models untuk stacking
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
    ))
]

# Meta learner
meta_learner = LogisticRegression(max_iter=1000)

# Stacking model
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5
)

# Model comparison
models = {
    'Decision Tree': DecisionTreeClassifier(
        random_state=42
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    'AdaBoost': AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    ),

    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),

    'Stacking': stacking
}

# Cross Validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, model in models.items():

    start_time = time.time()

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring='accuracy'
    )

    elapsed_time = time.time() - start_time

    results.append({
        'Model': name,
        'Mean Accuracy': scores.mean(),
        'Std': scores.std(),
        'Training Time (s)': elapsed_time
    })

# Hasil
results_df = pd.DataFrame(results).round(4)

print(results_df.to_string(index=False))

# Model terbaik
best_model = results_df.loc[
    results_df['Mean Accuracy'].idxmax(),
    'Model'
]

print(f"\nModel terbaik berdasarkan CV: {best_model}")