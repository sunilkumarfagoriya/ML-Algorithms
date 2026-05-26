import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def euclidean_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def get_neighbours(X, point_idx, eps):
    neighbour_points = list()

    for idx, point in enumerate(X):
        dist = euclidean_dist(point_idx, point)
        
        if dist <= eps:
            neighbour_points.append(idx)

    return neighbour_points


def dbscan(X, eps, min_pts):
    neighbours = list()

    for point in X:
        neighbours.append(get_neighbours(X, point, eps))

    point_type = list()

    for idx in range(len(neighbours)):
        if len(neighbours[idx]) >= min_pts:
            point_type.append('Core')
            continue
        
        check = False

        for point in neighbours[idx]:
            if len(neighbours[point]) >= min_pts:
                check = True
                break

        if check:
            point_type.append('Border')
        else:
            point_type.append('Noise')
    
    return neighbours, point_type


def rec(idx, cluster, cluster_i, neighbour, point_type):

    for point in neighbour[idx]:
        if(point_type[point] == 'Core' and cluster[point] == -1):
            cluster[point] = cluster_i
            rec(point, cluster, cluster_i, neighbour, point_type)
        elif(point_type[point] == 'Border' and cluster[point] == -1):
                cluster[point] = cluster_i


def asign_cluster(neighbour, point_type):
    cluster = [-1] * len(neighbour)
    cluster_i, i = 0, 0

    while(i < len(neighbour)):
        if point_type[i] == 'Core' and cluster[i] == -1:
            cluster[i] = cluster_i
            rec(i, cluster, cluster_i, neighbour, point_type)
            cluster_i += 1
        
        while(i < len(neighbour) and (point_type[i] != 'Core' or (cluster[i] != -1 and point_type[i] == 'Core'))):
            i += 1

    return cluster


data = pd.read_csv('C:/Users/Lenovo/Desktop/python librarys/DataPreprocessing/Mall_Customers.csv')

X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

min_pts = 2 * len(X[0])

k_dist = list()

for i in range(len(X)):
    dist = list()
    for j in range(len(X)):
        if(i != j):
            dist.append(euclidean_dist(X[i], X[j]))
    
    dist.sort()
    k_dist.append(dist[min_pts - 1])

k_dist.sort()
plt.figure(figsize = (10, 6))
plt.plot(k_dist)
plt.title('K-Distance Graph for finding epsilon')
plt.xlabel('Points')
plt.ylabel('K-th Nearest Distance')
plt.grid(True)
plt.show()

diffs = np.diff(k_dist)
elbow_idx = np.argmax(diffs)
eps = k_dist[elbow_idx]

neighbour, point_type = dbscan(X, eps, min_pts)

labels = asign_cluster(neighbour, point_type)
labels = np.array(labels)

plt.figure(figsize = (12, 5))

unique_labels = set(labels)

colors = ['red','blue','green','orange','purple','brown','pink']

for i in unique_labels:
    if(i == -1):
        point = X[labels == i]
        plt.scatter(point[:, 0], point[:, 1], c = 'black', marker = 'x', s = 100, label = 'Noise')
    else:
        point = X[labels == i]
        plt.scatter(point[:, 0], point[:, 1], c = colors[i % len(colors)], label = f'Cluster {i+1}')

plt.title('DBSCAN Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()







from sklearn.datasets import make_moons

# Irregular shape data
X_moons, _ = make_moons(n_samples=200, 
                         noise=0.1, 
                         random_state=42)

min_pts = 4

k_dist = []
for i in range(len(X_moons)):
    dist = []
    for j in range(len(X_moons)):
        if i != j:
            dist.append(euclidean_dist(X_moons[i], X_moons[j]))
    dist.sort()
    k_dist.append(dist[min_pts-1])

k_dist.sort()

diffs = np.diff(k_dist)
elbow_idx = np.argmax(diffs)
eps = k_dist[elbow_idx]

neighbour, point_type = dbscan(X_moons, eps, min_pts)
labels = np.array(asign_cluster(neighbour, point_type))

plt.figure(figsize=(10,6))
for i in set(labels):
    if i == -1:
        pts = X_moons[labels==-1]
        plt.scatter(pts[:,0], pts[:,1], 
                   c='black', marker='x', label='Noise')
    else:
        pts = X_moons[labels==i]
        plt.scatter(pts[:,0], pts[:,1], 
                   label=f'Cluster {i+1}')

plt.title('DBSCAN on Moon Dataset')
plt.legend()
plt.show()
