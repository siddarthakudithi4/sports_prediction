import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Load the dataset
file_path = 'IPL-Winner-Predictor-main/Updated_Second_Innings_Data.csv'

data = pd.read_csv(file_path)

# Convert 'wickets_remaining' to numeric
data['wickets_remaining'] = pd.to_numeric(data['wickets_remaining'], errors='coerce')

# Drop rows with missing values
data_cleaned = data[['total_runs', 'wickets_remaining']].dropna()

# Downsample the dataset further for faster computation
data_sampled = data_cleaned.sample(n=500, random_state=42).values

# --------------------- K-Means Clustering ---------------------
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def kmeans(data, k, max_iters=50):  # Reduced iterations
    centroids = data[np.random.choice(len(data), k, replace=False)]
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters if len(cluster) > 0]
        new_centroids = np.array(new_centroids)
        if np.allclose(centroids, new_centroids):  # Early stopping if centroids don't move
            break
        centroids = new_centroids
    return clusters, centroids

# --------------------- GMM (Gaussian Mixture Model) ---------------------
def gaussian_pdf(x, mean, cov):
    return multivariate_normal(mean=mean, cov=cov).pdf(x)

def gmm(data, k, max_iters=50):  # Reduced iterations
    n, d = data.shape
    means = data[np.random.choice(len(data), k, replace=False)]
    covariances = [np.cov(data.T) for _ in range(k)]
    pi = np.ones(k) / k
    likelihoods = np.zeros((n, k))

    for _ in range(max_iters):
        # E-step
        for i in range(k):
            likelihoods[:, i] = pi[i] * gaussian_pdf(data, means[i], covariances[i])
        
        responsibilities = likelihoods / likelihoods.sum(axis=1, keepdims=True)
        
        # M-step
        Nk = responsibilities.sum(axis=0)
        means = np.dot(responsibilities.T, data) / Nk[:, None]
        
        covariances = []
        for i in range(k):
            diff = data - means[i]
            weighted_diff = responsibilities[:, i][:, None] * diff
            cov_matrix = np.dot(weighted_diff.T, diff) / Nk[i]
            covariances.append(cov_matrix)
        
        pi = Nk / n

    return means, covariances, responsibilities

# --------------------- Hierarchical Clustering ---------------------
def hierarchical_clustering(data, k):
    clusters = [[point] for point in data]
    
    while len(clusters) > k:
        min_distance = float('inf')
        to_merge = (0, 0)
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = euclidean_distance(np.mean(clusters[i], axis=0), np.mean(clusters[j], axis=0))
                if dist < min_distance:
                    min_distance = dist
                    to_merge = (i, j)
        i, j = to_merge
        clusters[i].extend(clusters[j])
        clusters.pop(j)
    
    return clusters

# --------------------- Run and Plot Results ---------------------

# K-Means
kmeans_clusters, kmeans_centroids = kmeans(data_sampled, k=3)
kmeans_labels = np.concatenate([[i]*len(cluster) for i, cluster in enumerate(kmeans_clusters)])

# GMM
gmm_means, gmm_covariances, gmm_responsibilities = gmm(data_sampled, k=3)
gmm_labels = np.argmax(gmm_responsibilities, axis=1)

# Hierarchical Clustering
hierarchical_clusters = hierarchical_clustering(data_sampled, k=3)
hierarchical_labels = np.concatenate([[i]*len(cluster) for i, cluster in enumerate(hierarchical_clusters)])

# Plot the results for K-Means
plt.figure(figsize=(8, 6))
for i, cluster in enumerate(kmeans_clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i+1}')
plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Total Runs')
plt.ylabel('Wickets Remaining')
plt.legend()
plt.show()

# Plot the results for GMM
plt.figure(figsize=(8, 6))
plt.scatter(data_sampled[:, 0], data_sampled[:, 1], c=gmm_labels, cmap='viridis', s=50)
plt.title('GMM Clustering')
plt.xlabel('Total Runs')
plt.ylabel('Wickets Remaining')
plt.show()

# Plot the results for Hierarchical Clustering
plt.figure(figsize=(8, 6))
for i, cluster in enumerate(hierarchical_clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i+1}')
plt.title('Hierarchical Clustering')
plt.xlabel('Total Runs')
plt.ylabel('Wickets Remaining')
plt.legend()
plt.show()
