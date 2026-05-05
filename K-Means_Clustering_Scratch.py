import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def euclidean_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def find_distance(X, centroids):
    dist = []
    for i in X:
        point_dist = []
        
        for j in centroids:
            point_dist.append(euclidean_dist(i, j))
        
        dist.append(point_dist)
    
    return dist


def init_centroids(X, k):
    store_k = []
    store_k.append(X[np.random.randint(0, len(X))])

    for i in range(k-1):
        dist = find_distance(X, store_k)
        
        min_dist = np.min(dist, axis = 1)
        dist_prob = (min_dist ** 2)/np.sum(min_dist ** 2)

        idx = np.random.choice(len(X), p = dist_prob)
        next_k = X[idx]
        
        store_k.append(next_k)
    
    return store_k


def assign_clusters(X, centroids):
    dist = find_distance(X, centroids)
    min_dist = np.argmin(dist, axis = 1)
    return min_dist


def update_centroids(X, assignments, k):
    
    for i in range(len(k)):
        k[i] = X[assignments == i].mean(axis = 0)
    
    return k


def build_kmeans(X, k):

    while(1):
        assignments = assign_clusters(X, k)
        next_k = update_centroids(X, assignments, k)

        if np.allclose(k, next_k):
            break

        k = next_k
    
    return k


def wcss(X, assignments, centroids):
    css = 0

    for i in range(len(X)):
        css += (euclidean_dist(X[i], centroids[assignments[i]]) ** 2)
    
    return css


np.random.seed(42)

cluster1 = np.random.randn(50, 2) + [1, 1]
cluster2 = np.random.randn(50, 2) + [5, 5]
cluster3 = np.random.randn(50, 2) + [9, 1]

X = np.vstack([cluster1, cluster2, cluster3])

wcss_list = []

for k in range(1, 9):
    best_css = float('inf')

    for i in range(10):

        centroids = init_centroids(X, k)
        centroids = build_kmeans(X, centroids)
        assignments = assign_clusters(X, centroids)
        css = wcss(X, assignments, centroids)

        if(css < best_css):
            best_css = css

    wcss_list.append(best_css)

plt.plot(range(1, 9), wcss_list, 'bo-')
plt.xlabel('K')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()


sil_list = []

for k in range(2, 9):
    best_sil = -1

    for i in range(10):
        
        centroids = init_centroids(X, k)
        centroids = build_kmeans(X, centroids)
        assignments = assign_clusters(X, centroids)
        sil_score = silhouette_score(X, assignments)

        if(sil_score > best_sil):
            best_sil = sil_score
        
    sil_list.append(best_sil)

plt.plot(range(2, 9), sil_list, 'bo-')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()


idx = np.argmax(sil_list)

k = idx + 2

print(k)
