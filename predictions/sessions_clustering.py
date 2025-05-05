import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import seaborn as sns

mpl.rcParams.update({'font.size': 16})
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Charter', 'XCharter', 'Georgia', 'Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'

file_path = "data/combined_charging_sessions_prepro.csv"
df = pd.read_csv(file_path, parse_dates=['plug_in_datetime', 'plug_out_datetime'])

# Normalization

df['plug_in_time_norm'] = df['plug_in_time'] / 24

df['connected_duration'] = df['connected_duration'].clip(0, 48)  # Ensure max is 48
df['connected_duration_norm'] = df['connected_duration'] / 48

df['HBS'] = df['HBS'].clip(0, 196)
df['HBS_norm'] = df['HBS'] / 196

df['normalized_log_DBS'] = df['log_DBS'] / df['log_DBS'].max()
df = df[df['normalized_log_DBS'] > 0]

df = df[df['next_CBS'] > 0]

features = ['normalized_log_DBS', 'HBS_norm', 'connected_duration_norm', 'plug_in_time_norm']

# Number of clusters

# Store BIC scores for plotting
bic_scores = []
best_gmm = None
best_bic = np.inf
best_n = None

n_range = range(1, 15)

for n in n_range:
    print(f"Training GMM with {n} clusters")
    gmm = GaussianMixture(n_components=n, covariance_type='full', reg_covar=1e-6, random_state=42)
    gmm.fit(df[features])

    bic = gmm.bic(df[features])
    bic_scores.append(bic)

    if bic < best_bic:
        best_bic = bic
        best_gmm = gmm
        best_n = n

# Plot BIC scores
plt.figure(figsize=(10, 5))
plt.plot(n_range, bic_scores, marker='o', linestyle='-')
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('BIC Score', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.savefig("figures/cluster2_bic.pdf")

# Detect the "elbow point" where BIC improvement slows
knee_locator = KneeLocator(n_range, bic_scores, curve="convex", direction="decreasing")
optimal_n_knee = knee_locator.knee
print(f"Optimal number of clusters (elbow method): {optimal_n_knee}")

ari_matrix = []
cluster_stability = []

# ARI score
for n in range(optimal_n_knee - 3, optimal_n_knee + 3):
    print(f"Testing ARI with {n} clusters")
    gmm_test = GaussianMixture(n_components=n, covariance_type='full', reg_covar=1e-6, random_state=42)
    labels = gmm_test.fit_predict(df[features])

    # Calculate ARI score and append to the matrix
    if len(cluster_stability) > 0:
        ari_score = adjusted_rand_score(cluster_stability[-1], labels)
        print(f"ARI score between {n - 1} and {n} clusters: {ari_score:.4f}")
    else:
        ari_score = -1  # No previous labels to compare, use -1 as a placeholder
        print(f"ARI score between {n - 1} and {n} clusters: negative")

    # Add the ARI score to the matrix
    if len(ari_matrix) > 0:
        # Ensure the row length matches by padding with NaN if needed
        while len(ari_matrix[-1]) < len(cluster_stability):
            ari_matrix[-1].append(np.nan)
        ari_matrix[-1].append(ari_score)
    else:
        ari_matrix.append([ari_score])

    cluster_stability.append(labels)

n_components = optimal_n_knee + 1
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
df['SessionCluster'] = gmm.fit_predict(df[features])

probs = gmm.predict_proba(df[features])
df['Max_Prob'] = probs.max(axis=1)
df['SessionCluster'] = np.where(df['Max_Prob'] < 0.7, 'Noise', df['SessionCluster'])

cluster_distribution = df['SessionCluster'].value_counts()

noise_count = cluster_distribution['Noise']
noise_proportion = noise_count / len(df)
print(f"Noise proportion: {noise_proportion}")

df.to_csv("data/combined_charging_sessions_clustered.csv", index=False)

variables = ["plug_in_time", "connected_duration", "HBS", "DBS"]
titles = ["Plug-in Time", "Connected Duration", "HBS", "DBS"]

for cluster in df["SessionCluster"].unique():
    cluster_data = df[df["SessionCluster"] == cluster]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, var, title in zip(axes.flatten(), variables, titles):
        sns.histplot(cluster_data[var], bins=20, kde=True, ax=ax)
        ax.set_title(f"{title} - Cluster {cluster}")
        ax.set_xlabel(var)
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"figures/session_cluster/cluster_{cluster}_distributions.pdf")
