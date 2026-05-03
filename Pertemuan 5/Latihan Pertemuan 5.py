import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =============================
# k-NN from scratch
# =============================
class KNN_FromScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # hitung jarak ke semua data training
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

        # ambil k tetangga terdekat
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # voting mayoritas
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# =============================
# Load & preprocessing data
# =============================
iris = load_iris()
X = iris.data
y = iris.target

# normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# =============================
# Eksperimen berbagai nilai k
# =============================
k_values = [1, 3, 5, 7, 9, 11]

print("Perbandingan Akurasi k-NN (Scratch vs Scikit-Learn)\n")

for k in k_values:
    # From scratch
    knn_scratch = KNN_FromScratch(k=k)
    knn_scratch.fit(X_train, y_train)
    y_pred_scratch = knn_scratch.predict(X_test)
    acc_scratch = accuracy_score(y_test, y_pred_scratch)

    # Scikit-learn
    knn_sklearn = KNeighborsClassifier(n_neighbors=k)
    knn_sklearn.fit(X_train, y_train)
    y_pred_sklearn = knn_sklearn.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

    print(f"k = {k}")
    print(f"  Scratch      : {acc_scratch:.4f}")
    print(f"  Scikit-learn : {acc_sklearn:.4f}\n")

# =============================
# Analisis sederhana
# =============================
print("Analisis:")
print("- k kecil (1-3): akurasi tinggi tapi rawan overfitting")
print("- k sedang (5-7): biasanya paling stabil")
print("- k besar (9-11): mulai menurun (underfitting)")