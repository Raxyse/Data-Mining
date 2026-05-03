import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cityblock, minkowski
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# EUCLIDEAN DAN MANHATTAN
P = np.array([1, 2, 3])
Q = np.array([4, 5, 6])

euclidean_distance = np.sqrt(np.sum((P - Q) ** 2))
manhattan_distance = np.sum(np.abs(P - Q))

print("EUCLIDEAN DAN MANHATTAN")
print(f"Euclidean Distance : {euclidean_distance:.2f}")
print(f"Manhattan Distance : {manhattan_distance:.2f}")

# SMC DAN JACCARD
X_binary = np.array([1, 0, 1, 0, 1, 1])
Y_binary = np.array([1, 1, 0, 0, 1, 0])

m11 = np.sum((X_binary == 1) & (Y_binary == 1))
m00 = np.sum((X_binary == 0) & (Y_binary == 0))
m10 = np.sum((X_binary == 1) & (Y_binary == 0))
m01 = np.sum((X_binary == 0) & (Y_binary == 1))

smc = (m11 + m00) / (m11 + m00 + m10 + m01)
jaccard = m11 / (m11 + m10 + m01)

print("\nSMC DAN JACCARD")
print(f"SMC : {smc:.2f}")
print(f"Jaccard : {jaccard:.2f}")


# IMPLEMENTASI JARAK NUMERIK PADA DATASET IRIS
# Load dataset iris
iris = load_iris()
iris_data = iris.data
feature_names = iris.feature_names

# Ubah ke DataFrame
df = pd.DataFrame(iris_data, columns=feature_names)

print("\n==DATASET IRIS==")
print(df.head())

# NORMALISASI DATA
scaler = MinMaxScaler()
iris_normalized = scaler.fit_transform(iris_data)

# Pakai data hasil normalisasi
data = iris_normalized

print("\nDATA SETELAH NORMALISASI")
print(pd.DataFrame(data, columns=feature_names).head())

# CONTOH PERHITUNGAN JARAK
A = data[0]
B = data[1]

euc_dist = euclidean(A, B)
man_dist = cityblock(A, B)
min_dist = minkowski(A, B, p=3)

print("\nCONTOH PERHITUNGAN DATA KE-0 DAN DATA KE-1")
print(f"Euclidean Distance : {euc_dist:.4f}")
print(f"Manhattan Distance : {man_dist:.4f}")
print(f"Minkowski Distance : {min_dist:.4f}")

# LOOPING JARAK 5 DATA PERTAMA
print("\n==PERHITUNGAN JARAK 5 DATA PERTAMA==")

for i in range(5):
    for j in range(i + 1, 5):
        euc_dist = euclidean(data[i], data[j])
        man_dist = cityblock(data[i], data[j])
        min_dist = minkowski(data[i], data[j], p=3)

        print(f"\nData ke-{i} dengan Data ke-{j}")
        print(f"Euclidean Distance : {euc_dist:.4f}")
        print(f"Manhattan Distance : {man_dist:.4f}")
        print(f"Minkowski Distance : {min_dist:.4f}")

# MATRIKS JARAK
n = len(data)

euclidean_matrix = np.zeros((n, n))
manhattan_matrix = np.zeros((n, n))
minkowski_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        euclidean_matrix[i][j] = euclidean(data[i], data[j])
        manhattan_matrix[i][j] = cityblock(data[i], data[j])
        minkowski_matrix[i][j] = minkowski(data[i], data[j], p=3)

print("\n==MATRIKS JARAK EUCLIDEAN (5x5)==")
print(pd.DataFrame(euclidean_matrix[:5, :5]))

print("\n==MATRIKS JARAK MANHATTAN (5x5)==")
print(pd.DataFrame(manhattan_matrix[:5, :5]))

print("\n==MATRIKS JARAK MINKOWSKI (5x5)==")
print(pd.DataFrame(minkowski_matrix[:5, :5]))

# COSINE SIMILARITY
documents = [
    "data science is fun",
    "data mining is fun",
    "machine learning is cool"
]

vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(documents)

print("\nVOCABULARY")
print(vectorizer.get_feature_names_out())

print("\nDOCUMENT TERM MATRIX")
print(X_text.toarray())

cos_sim = cosine_similarity(X_text)

print("\nCOSINE SIMILARITY MATRIX")
print(cos_sim)

print("\nCosine Similarity D1-D2 :", cos_sim[0][1])
print("Cosine Similarity D1-D3 :", cos_sim[0][2])