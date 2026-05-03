import numpy as np
from collections import Counter

def knn_predict(x_train, y_train, x_test, k=3):
    """Prediksi dengan k-NN menggunakan Euclidean distance"""
    predictions = []
    for test_point in x_test:
        # Hitung jarak ke semua training point
        distances = []
        for i, train_point in enumerate(x_train):
            dist = np.sqrt(np.sum((test_point - train_point)**2))
            distances.append((dist, y_train[i]))
        # Urutkan berdasarkan jarak
        distances.sort(key=lambda x: x[0])
        # Ambil k tetangga terdekat
        k_neighbors = distances[:k]
        k_labels = [label for _, label in k_neighbors]
        # Majority voting
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)

    return np.array(predictions)

# Contoh: klasifikasi bunga iris sederhana
x_train = np.array([[5.1, 3.5], [4.9, 3.0], [6.2, 3.4], [5.9, 3.0]])
y_train = np.array([0, 0, 1, 1]) # 0=setosa, 1=versicolor

x_test = np.array([[5.0, 3.2], [6.0, 3.2]])

predictions = knn_predict(x_train, y_train, x_test, k=3)
print(f"Hasil prediksi: {predictions}")
# Output: [0 1]