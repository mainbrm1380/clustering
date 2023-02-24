import pandas as pd
import numpy as np


df = pd.read_csv("iris.csv", header=None)

main = df[[0,1,2,3]]
main.columns = ["sepal length", "sepal width", "pedal length", "pedal width"]

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class Kmeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

        self.clusters = [[]for _ in range (self.k)]
        self.centroids = []

    def predict(self, x):
        self.x = x
        self.n_samples, self.n_features = np.shape(x) 
        
        random_sample_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.x[idx] for idx in random_sample_idxs]
        
        for _ in range(self.max_iters):
            self.clusters = self.createClusters(self.centroids)
            centroids_old = self.centroids
            self.centroids = self.getCentroids(self.clusters)
            if self.isConverged(centroids_old, self.centroids):
                break

        return self.clusters

    def createClusters(self, centeroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.x):
            centroid_idx = self.closestCentroid(sample, centeroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def closestCentroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def getCentroids(self, clusters):
        centeroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            clusterMean = np.mean(self.x[cluster], axis=0)
            centeroids[cluster_idx] = clusterMean
        return centeroids

    def isConverged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0

x = np.array(main)

k = Kmeans()
clusteres = k.predict(x)
print(f"{clusteres}\n")
print(pd.DataFrame(clusteres))