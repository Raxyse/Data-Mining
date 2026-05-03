import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, roc_auc_score)

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
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(probability=True, random_state=42)
}

# 5. Evaluasi dengan Stratified 10-Fold CV
print("="*50)
print(" STRATIFIED 10-FOLD CROSS VALIDATION ")
print("="*50)

for name, model in models.items():
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    prec = cross_val_score(model, X, y, cv=cv, scoring='precision')
    rec = cross_val_score(model, X, y, cv=cv, scoring='recall')
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"{name}: Acc={acc.mean():.3f}, Prec={prec.mean():.3f}, Rec={rec.mean():.3f}, F1={f1.mean():.3f}, AUC={auc.mean():.3f}")

# 6. Plot ROC curve semua model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
plt.figure(figsize=(10, 8))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
