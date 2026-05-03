import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset Titanic
df = pd.read_csv(r"C:\Users\ThinkPad L14\Documents\train.csv")

# 2. Target dan fitur
y = df["Survived"]
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

# 3. Preprocessing
X = pd.get_dummies(X, columns=["Sex", "Embarked"], drop_first=True)
X["Age"].fillna(X["Age"].median(), inplace=True)
X["Fare"].fillna(X["Fare"].median(), inplace=True)

# 4. Definisikan model
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(probability=True, random_state=42)
}

# 5. Plot learning curve untuk tiap model
plt.figure(figsize=(12, 8))

for name, model in models.items():
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=1   # aman di Windows + Python 3.13
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', label=f'{name} - Train')
    plt.plot(train_sizes, test_mean, 'o--', label=f'{name} - CV')

plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Learning Curve Comparison')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.show()

# 6. Diagnostik sederhana untuk tiap model
for name, model in models.items():
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=1
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    gap = train_mean[-1] - test_mean[-1]
    if gap > 0.1:
        status = "OVERFITTING"
    elif gap < -0.05:
        status = "UNDERFITTING"
    else:
        status = "GOOD"
    print(f"{name}: Train={train_mean[-1]:.3f}, CV={test_mean[-1]:.3f}, Gap={gap:.3f} → {status}")
