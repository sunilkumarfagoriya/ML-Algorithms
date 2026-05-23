import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

def euclidean_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def distance_matrix(X):
    matrix = list()
    n = len(X)
    
    for i in range(n):
        row_dist = list()
        
        for j in range(n):
            row_dist.append(float(euclidean_dist(X[i], X[j])))
        matrix.append(row_dist)
    
    return matrix


def find_closest(matrix):
    min = 100000000
    row, col = 0, 0
    
    for i in range(len(matrix)):
        for j in range(i):
            if(min > matrix[i][j]):
                min = matrix[i][j]
                row, col = i, j
    
    return row, col


def merge_clusters(matrix, i, j):
    new_row = list()

    for k in range(len(matrix)):
        if k != i and k != j:
            avg = (matrix[i][k] + matrix[j][k]) / 2
            new_row.append(avg)
    
    new_row.append(0)


    if(i > j):
        del matrix[i], matrix[j]
    else:
        del matrix[j], matrix[i]

    for k in range(len(matrix)):
        if(i > j):
            del matrix[k][i], matrix[k][j]
        else:
            del matrix[k][j], matrix[k][i]


    matrix.append(new_row)

    for k in range(len(new_row) - 1):
        matrix[k].append(new_row[k])
    

    return matrix


def build_tree(X):
    matrix = distance_matrix(X)
    history = []
    clusters = list(range(len(X)))

    while(len(matrix) != 1):
        i, j = find_closest(matrix)
        history.append([clusters[i], clusters[j]])

        clusters.append([clusters[i], clusters[j]])
        del clusters[max(i, j)]
        del clusters[min(i, j)]

        matrix = merge_clusters(matrix, i, j)

    return history


mall_data = pd.read_csv('C:/Users/Lenovo/Desktop/python librarys/DataPreprocessing/Mall_Customers.csv')

X = mall_data[['Annual Income (k$)', 'Spending Score (1-100)']].values

cluster_data = build_tree(X)

linked = linkage(X, method = 'ward')

plt.figure(figsize = (12, 6))

dendrogram(linked)

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

model = AgglomerativeClustering(n_clusters = 5)
labels = model.fit_predict(X)

plt.figure(figsize = (10, 6))
colors = ['red','blue','green','orange','purple']

for i in range(5):
    points = X[labels == i]
    plt.scatter(points[:, 0], points[:, 1], c = colors[i], label = f'Cluster {i+1}')

plt.title('Hierarchical Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.tight_layout()
plt.show()

