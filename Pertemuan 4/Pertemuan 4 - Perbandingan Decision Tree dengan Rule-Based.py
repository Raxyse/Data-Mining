import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# Load dataset Titanic (CSV lokal dari Kaggle)
# -----------------------------
# Ganti path sesuai lokasi file train.csv di komputer kamu
titanic = pd.read_csv(r"C:\Users\ThinkPad L14\Documents\train.csv")

# Pilih kolom yang relevan
titanic = titanic[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]]
titanic = titanic.dropna()
titanic["Sex"] = titanic["Sex"].map({"male": 0, "female": 1})

X = titanic.drop("Survived", axis=1)
y = titanic["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# -----------------------------
# 1. Single Decision Tree
# -----------------------------
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# -----------------------------
# 2. Random Forest
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# -----------------------------
# 3. Gradient Boosting
# -----------------------------
gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

# -----------------------------
# 4. Stacking (minimal 3 base models)
# -----------------------------
base_models = [
    ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
    ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=5))
]
meta_learner = LogisticRegression(max_iter=1000)

stacking = StackingClassifier(estimators=base_models, final_estimator=meta_learner, cv=5)
stacking.fit(X_train, y_train)
stacking_pred = stacking.predict(X_test)

# -----------------------------
# Evaluasi semua model
# -----------------------------
models = {
    "Decision Tree": dt_pred,
    "Random Forest": rf_pred,
    "Gradient Boosting": gb_pred,
    "Stacking": stacking_pred
}

results = []
for name, pred in models.items():
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    results.append([name, acc, prec, rec, f1])

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"])
print(results_df.round(4))

# -----------------------------
# Classification Report contoh
# -----------------------------
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_pred))
