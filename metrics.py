import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split


def RankMe(embedding):
    s = np.linalg.svd(embedding, compute_uv=False)
    p = s / np.abs(s.sum())
    entropy = -(p*np.log(p)).sum()
    rankme = np.exp(entropy)
    return rankme
def get_eigenspectrum(activations_np,max_eigenvals=2048):
    feats = activations_np.reshape(activations_np.shape[0],-1)
    feats_center = feats - feats.mean(axis=0)
    pca = PCA(n_components=min(max_eigenvals, feats_center.shape[0], feats_center.shape[1]), svd_solver='full')
    pca.fit(feats_center)
    eigenspectrum = pca.explained_variance_ratio_
    return eigenspectrum

def fit_powerlaw(arr, start, end):
    x_range = np.arange(start, end + 1).astype(int)
    y_range = arr[x_range - 1]  # because the first eigenvalue is at index 0, so eigenval_{start} is at index (start-1)
    reg = LinearRegression().fit(np.log(x_range).reshape(-1, 1), np.log(y_range).reshape(-1, 1))
    y_pred = np.exp(reg.coef_ * np.log(x_range).reshape(-1, 1) + reg.intercept_)
    return -reg.coef_[0][0], x_range, y_pred

def stringer_get_powerlaw(ss, trange):
    # COPIED FROM Stringer+Pachitariu 2018b github repo! (https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/utils.py)
    ''' fit exponent to variance curve'''
    logss = np.log(np.abs(ss))
    y = logss[trange][:, np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:, np.newaxis], np.ones((nt, 1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:, np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:, np.newaxis], np.ones((ss.size, 1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    max_range = 500 if len(ss) >= 512 else len(
        ss) - 10  # subtracting 10 here arbitrarily because we want to avoid the last tail!
    fit_R2 = r2_score(y_true=logss[trange[0]:max_range], y_pred=np.log(np.abs(ypred))[trange[0]:max_range])
    try:
        fit_R2_100 = r2_score(y_true=logss[trange[0]:100], y_pred=np.log(np.abs(ypred))[trange[0]:100])
    except:
        fit_R2_100 = None
    return alpha, ypred, fit_R2, fit_R2_100




class ClusteringAnalyzer:

    def __init__(self, num_clusters, embeddings, num_neighbors=1) -> None:
        """
        Initialize the ClusterAnalyzer.
        Args:
            num_clusters (int): The number of clusters to be formed.
            embeddings (numpy.ndarray): The embeddings of the samples.
            num_neighbors (int, optional): The number of neighbors to consider in the online KNN classification. Default is 1.
        """
        self.num_clusters = num_clusters
        self.embeddings = embeddings
        self.num_neighbors = num_neighbors
        self.cluster_assignments = self.kmeans_clustering()
        self.updated_labels = None

    def kmeans_clustering(self):
        """
        Perform K-means clustering on the embeddings.
        Returns:
            numpy.ndarray: The cluster labels assigned to each sample.
        """
        kmeans = KMeans(n_clusters=self.num_clusters)
        self.cluster_assignments = kmeans.fit_predict(self.embeddings)
        return self.cluster_assignments

    def online_knn_classification(self, cluster_labels):
        """
        Perform online K-nearest neighbors (KNN) classification.
        Args:
            labels (numpy.ndarray): The initial cluster labels assigned to each sample.
        Returns:
            numpy.ndarray: The updated cluster labels after online KNN classification.
        """
        labels = np.zeros_like(cluster_labels)

        knn = KNeighborsClassifier(n_neighbors=self.num_neighbors,metric='cosine')
        X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.cluster_assignments, test_size=0.5, random_state=42)

        knn.fit(X_train, y_train)
        labels = knn.predict(X_test)
        same_label_count = sum(y_test[i] == labels[i] for i in range(len(labels)))
        cluster_learnability = same_label_count / len(labels)

        return cluster_learnability #/2


    def compute_cluster_learnability(self):
        """
        Compute the cluster learnability metric based on https://arxiv.org/abs/2206.01251.
        Returns:
            float: The cluster learnability, which represents the proportion of samples with unchanged cluster labels.
        """
        cluster_labels = self.kmeans_clustering()
        #self.updated_labels = self.online_knn_classification(cluster_labels)
        cluster_learnability= self.online_knn_classification(cluster_labels)
        # same_label_count = sum(self.cluster_assignments[i] == self.updated_labels[i] for i in range(len(self.embeddings)))
        # cluster_learnability = same_label_count / len(self.embeddings)
        return cluster_learnability
    def compute_silhouette_coefficient(self):
        """
        Compute the Silhouette Coefficient of embeddings given the corresponding cluster labels.

        Returns:
            float: The Silhouette Coefficient between -1 and 1.
        """
        labels = self.cluster_assignments
        distances = pairwise_distances(self.embeddings)
        num_samples = len(self.embeddings)
        silhouette_values = np.zeros(num_samples)

        for i in range(num_samples):
            # Calculate the intra-cluster distance for sample i
            intra_cluster_distances = distances[i][labels == labels[i]]
            avg_intra_cluster_distance = np.mean(intra_cluster_distances)

            # Calculate the nearest-cluster distance for sample i
            nearest_cluster_distances = distances[i][labels != labels[i]]
            avg_nearest_cluster_distance = np.mean(nearest_cluster_distances)

            # Calculate the Silhouette Coefficient for sample i
            silhouette_coefficient = (avg_nearest_cluster_distance - avg_intra_cluster_distance) / max(avg_intra_cluster_distance, avg_nearest_cluster_distance)

            silhouette_values[i] = silhouette_coefficient

        mean_silhouette_coefficient = np.mean(silhouette_values)

        return mean_silhouette_coefficient