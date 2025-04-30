import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import fcluster, linkage

df = pd.read_csv("data/combined_charging_sessions_clustered.csv", parse_dates=['plug_in_datetime', 'plug_out_datetime'])

# Portfolios
portfolio = df.groupby(["user_id", "SessionCluster"]).size().unstack(fill_value=0)
portfolio = portfolio.div(portfolio.sum(axis=1), axis=0)  # Convert counts to ratios

portfolio = portfolio.reset_index()

df_no_noise = df[df['SessionCluster'] != 'Noise']
user_sessions = df_no_noise.groupby('user_id').size()

users_with_enough_sessions = user_sessions[user_sessions >= 40].index
filtered_df = df[df['user_id'].isin(users_with_enough_sessions)]

filtered_df = filtered_df[filtered_df['SessionCluster'] != 'Noise']

users_in_df = filtered_df['user_id'].unique()

portfolio = portfolio[portfolio['user_id'].isin(users_in_df)]


# Voting for the pest method with the best k
#
# 4 methods :


def kmeans_clustering(n_clusters, data):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(data)


def pam_clustering(n_clusters, data):
    kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean', random_state=42)
    return kmedoids.fit_predict(data)


def hierarchical_ward_clustering(n_clusters, data):
    linkage_matrix = linkage(data, method='ward')
    return fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')


def hierarchical_complete_clustering(n_clusters, data):
    linkage_matrix = linkage(data, method='complete')
    return fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')


# Quality metrics (8)


def calinski_harabasz_index(data, labels):
    return calinski_harabasz_score(data, labels)


def davies_bouldin_index(data, labels):
    return davies_bouldin_score(data, labels)


def silhouette_index(data, labels):
    return silhouette_score(data, labels)


def duda_index(data, labels):
    clusters = np.unique(labels)
    score = 0
    for cluster in clusters:
        cluster_points = data[labels == cluster]
        centroid = np.mean(cluster_points, axis=0)
        intra_dist = np.mean(pairwise_distances(cluster_points, [centroid]))
        inter_dist = np.mean(pairwise_distances(cluster_points, data[labels != cluster]))
        score += intra_dist / inter_dist
    return score / len(clusters)


def pseudot2_index(data, labels):
    clusters = np.unique(labels)
    score = 0
    for cluster in clusters:
        cluster_points = data[labels == cluster]
        mean = np.mean(cluster_points, axis=0)
        dist = np.mean(pairwise_distances(cluster_points, [mean]))
        score += dist
    return score / len(clusters)


def c_index(data, labels):
    clusters = np.unique(labels)
    score = 0
    for cluster in clusters:
        cluster_points = data[labels == cluster]
        dist = np.mean(pairwise_distances(cluster_points))
        score += dist
    return score / len(clusters)


def gamma_index(data, labels):
    clusters = np.unique(labels)
    score = 0
    for cluster in clusters:
        cluster_points = data[labels == cluster]
        dist = np.mean(pairwise_distances(cluster_points))
        score += dist
    return score / len(clusters)


def beale_index(data, labels):
    clusters = np.unique(labels)
    score = 0
    for cluster in clusters:
        cluster_points = data[labels == cluster]
        dist = np.mean(pairwise_distances(cluster_points))
        score += dist
    return score / len(clusters)


# Evaluate from 2 to 15 k

methods = {
    "KMeans": kmeans_clustering,
    "PAM": pam_clustering,
    "Hierarchical Ward": hierarchical_ward_clustering,
    "Hierarchical Complete": hierarchical_complete_clustering
}

# Quality metrics list
quality_metrics = [
    ('Calinski-Harabasz', calinski_harabasz_index),
    ('Davies-Bouldin', davies_bouldin_index),
    ('Silhouette', silhouette_index),
    ('Duda', duda_index),
    ('Pseudot2', pseudot2_index),
    ('C-Index', c_index),
    ('Gamma', gamma_index),
    ('Beale', beale_index)
]

scaled_data = portfolio.drop('user_id', axis=1)

votes = {method: {k: 0 for k in range(2, 16)} for method in methods.keys()}

for method_name, clustering_function in methods.items():
    for k in range(2, 16):

        cluster_labels = clustering_function(k, scaled_data)

        for metric_name, metric_func in quality_metrics:
            score = metric_func(scaled_data, cluster_labels)
            if metric_name in ['Calinski-Harabasz', 'Silhouette']:
                if score == max([metric_func(scaled_data, clustering_function(k, scaled_data)) for k in range(2, 16)]):
                    votes[method_name][k] += 1
            elif metric_name in ['Davies-Bouldin', 'Duda', 'Pseudot2', 'C-Index', 'Gamma', 'Beale']:  # Lower is better
                if score == min([metric_func(scaled_data, clustering_function(k, scaled_data)) for k in range(2, 16)]):
                    votes[method_name][k] += 1

for method_name, method_votes in votes.items():
    print(f"Method: {method_name}")
    most_voted_k = max(method_votes, key=method_votes.get)
    print(f"Most voted k for {method_name}: {most_voted_k} with {method_votes[most_voted_k]} votes\n")

votes_df = pd.DataFrame(votes)
votes_df = votes_df.fillna(0)

total_votes = votes_df.sum(axis=1)

total_votes[total_votes == 0] = 1

percentages = (votes_df.T / total_votes).T * 100

# Plot
percentages.plot(kind='bar', stacked=False, colormap='Set2', width=0.8)

# Labels and formatting
plt.xlabel('k (Number of Clusters)', fontsize=12)
plt.ylabel('Percentage of Votes (%)', fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(fontsize=10)

# Legend
plt.legend(title="Clustering Method", loc='upper left', fontsize=10)

plt.savefig("figures/votes_portfolios.pdf")

method_votes = votes["KMeans"]
best_k = max(method_votes, key=method_votes.get)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
portfolio['cluster'] = kmeans.fit_predict(scaled_data)

df = df.merge(portfolio[['user_id', 'cluster']], on='user_id', how='left')

df = df.rename(columns={'cluster': 'user_cluster'})
df.drop(columns=['Unnamed: 0'], inplace=True)
df.to_csv("data/cluster2GM_final.csv")
