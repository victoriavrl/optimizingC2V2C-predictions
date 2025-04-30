import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("../data/combined_charging_sessions.csv", parse_dates=['plug_in_datetime', 'plug_out_datetime'])
unique_users = df['user_id'].unique()
# Filter based on energy session
df = df[df["total_energy_grid"] <= 120]

# Filter based on connection time
df = df[df["connected_duration"] <= 50]


# Function to get KDE or fallback value
def get_kde_or_fallback(data, fallback=1):
    if len(data) > 1:
        return gaussian_kde(data)
    elif len(data) == 1:
        return lambda x: data.iloc[0]  # Return a constant function
    else:
        print("fallback!")
        return lambda x: fallback  # Default fallback value


def f1_get_num_sessions(cluster, weekday):
    cluster_data = train_df[(train_df['cluster_step1'] == cluster) & (train_df['weekday'] == weekday)]
    freq_values = cluster_data['freq'].dropna().values * cluster_data['periodDays'].values

    if len(freq_values) > 1:
        freq_dist = gaussian_kde(freq_values)
        sampled_value = int(max(1, freq_dist.resample(1)[0].item()))  # Extract scalar
    elif len(freq_values) == 1:
        sampled_value = int(max(1, freq_values[0]))
    else:
        sampled_value = 1  # Default to at least 1 session

    return sampled_value


def f2_get_plugin_plugout_times(cluster):
    cluster_data = train_df[train_df['cluster_step1'] == cluster]

    plugin_times = cluster_data['plug_in_time']
    plugout_times = cluster_data['plug_out_time']

    kde_plugin_time = get_kde_or_fallback(plugin_times, fallback=np.median(plugin_times))
    kde_plugout_time = get_kde_or_fallback(plugout_times, fallback=np.median(plugout_times))

    sampled_plugin_time = kde_plugin_time.resample(1)[0].item()  # Extract scalar
    sampled_plugout_time = kde_plugout_time.resample(1)[0].item()  # Extract scalar

    # Ensure plug-out time is always after plug-in time
    if sampled_plugout_time <= sampled_plugin_time:
        sampled_plugout_time = sampled_plugin_time

    return int(max(0, sampled_plugin_time)), int(max(0, sampled_plugout_time))


def f3_get_energy_demand(cluster):
    cluster_data = train_df[train_df['cluster_step1'] == cluster]
    kde_energy = get_kde_or_fallback(cluster_data['total_energy_grid'],
                                     fallback=np.median(cluster_data['total_energy_grid']))

    return int(max(0, kde_energy.resample(1)[0].item()))  # Extract scalar


def chi_square_distance(real_data, synthetic_data, bins=50):
    real_hist, bin_edges = np.histogram(real_data, bins=bins, density=True)
    synthetic_hist, _ = np.histogram(synthetic_data, bins=bin_edges, density=True)

    chi_sq = 0.5 * np.sum((real_hist - synthetic_hist) ** 2 / (real_hist + synthetic_hist + 1e-10))
    return chi_sq


user_counts = [50, 100, 200, 500, 1000]
user_sessions = df.groupby('user_id').size()
# Store results for plotting later
inertia_by_user_count = {}
optimal_k_user_count = {}
n_range = range(2, 30)

features_step1 = ["mean_plugin_time", "mean_connection_time", "mean_energy_session"]

# Loop through each subset size
for count in user_counts:
    selected_users = np.random.choice(unique_users, size=count, replace=False)  # Random selection
    filtered_df = df[df['user_id'].isin(selected_users)].copy()
    train_df, val_df = train_test_split(filtered_df, test_size=0.1, random_state=42)

    theoretical_sessions = train_df.groupby("user_id").agg(
        mean_plugin_time=("plug_in_time", "mean"),
        mean_connection_time=("connected_duration", "mean"),
        mean_energy_session=("total_energy_grid", "mean")
    ).reset_index()

    # Frequency of charging
    # Compute frequency of charging: sessions per day
    last_day = train_df["plug_out_datetime"].max()

    charging_behavior = train_df.groupby("user_id").agg(
        numSessions=("plug_in_datetime", "count"),  # Count number of sessions
        firstSession=("plug_in_datetime", "min")  # Find the first session
    ).reset_index()

    # Ensure 'firstSession' and 'last_day' are datetime types
    charging_behavior["firstSession"] = pd.to_datetime(charging_behavior["firstSession"])

    # Compute the period in days since the first session
    charging_behavior["periodDays"] = (last_day - charging_behavior["firstSession"]).dt.days + 1

    # Compute frequency of charging: sessions per day
    charging_behavior["freq"] = charging_behavior["numSessions"] / charging_behavior["periodDays"]

    final_dataset = pd.merge(theoretical_sessions, charging_behavior[["user_id", "freq"]], on="user_id", how="inner")

    scaler = StandardScaler()
    df_scaled_step1 = scaler.fit_transform(final_dataset[features_step1])

    # Compute inertia (sum of squared distances) for different k values
    inertia = []
    k_values = range(2, 20)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled_step1)
        inertia.append(kmeans.inertia_)

    # Find the elbow point
    knee_locator = KneeLocator(k_values, inertia, curve="convex", direction="decreasing")
    optimal_k1 = knee_locator.elbow

    print(f"Optimal number of clusters ofr {count} users: {optimal_k1}")

    optimal_k_user_count[count] = optimal_k1
    inertia_by_user_count[count] = inertia

    kmeans_step1 = KMeans(n_clusters=optimal_k1, random_state=42, n_init=10)
    final_dataset["cluster_step1"] = kmeans_step1.fit_predict(df_scaled_step1)
    subclusters = {}

    for cluster in final_dataset["cluster_step1"].unique():
        print(f"Sub-Clustering for Cluster {cluster}...")

        # Select only drivers from this cluster
        df_cluster = final_dataset[final_dataset["cluster_step1"] == cluster].copy()

        # Sub-clustering only if multiple users exist
        if len(df_cluster) > 1:
            # Normalize only the "freq" feature
            scaler_freq = StandardScaler()
            df_cluster_scaled = scaler_freq.fit_transform(df_cluster[["freq"]])

            # Compute variance of the scaled feature
            freq_variance = np.var(df_cluster_scaled)

            # Find optimal k for sub-clustering
            inertia = []
            k_values = range(2, 5)

            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(df_cluster_scaled)
                inertia.append(kmeans.inertia_)

            # Find elbow
            knee_locator = KneeLocator(k_values, inertia, curve="convex", direction="decreasing")
            optimal_k2 = knee_locator.elbow if knee_locator.elbow else 2

            # Apply sub-clustering only if variance is significant
            if freq_variance > 0.01 and optimal_k2 > 1:  # Adjust threshold as needed
                kmeans_step2 = KMeans(n_clusters=optimal_k2, random_state=42, n_init=10)
                final_dataset.loc[final_dataset["cluster_step1"] == cluster, "subcluster"] = kmeans_step2.fit_predict(
                    df_cluster_scaled)

                subclusters[cluster] = optimal_k2
                print(f"Found {optimal_k2} sub-clusters.")
            else:
                print(f"Skipping sub-clustering for Cluster {cluster} due to low variance.")
                final_dataset.loc[final_dataset["cluster_step1"] == cluster, "subcluster"] = -1

    original_dataset = train_df
    merged_dataset = original_dataset.merge(final_dataset, on='user_id', how='left')
    merged_dataset['weekday'] = merged_dataset['plug_in_datetime'].dt.weekday
    merged_dataset['is_weekend'] = merged_dataset['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    clusters = merged_dataset['cluster_step1'].unique()
    train_df = merged_dataset
    train_df = train_df.merge(charging_behavior[['user_id', 'periodDays']], on='user_id', how='left')
    timestamp = (val_df['plug_out_datetime'].max() - val_df['plug_in_datetime'].min()).total_seconds() / 3600
    synthetic_sessions = []
    for cluster in train_df['cluster_step1'].unique():
        for weekday in range(7):
            num_sessions = f1_get_num_sessions(cluster, weekday)
            for _ in range(num_sessions):
                plug_in_time, plug_out_time = f2_get_plugin_plugout_times(cluster)
                energy_needed = f3_get_energy_demand(cluster)
                synthetic_sessions.append({
                    "Cluster": cluster,
                    "plug_in_time": plug_in_time,
                    "plug_out_time": plug_out_time,
                    "total_energy_grid": energy_needed
                })

    synthetic_dataset = pd.DataFrame(synthetic_sessions)
    synthetic_dataset['connected_duration'] = synthetic_dataset['plug_out_time'] - synthetic_dataset['plug_in_time']

    features = ["plug_in_time", "connected_duration", "total_energy_grid"]
    chi_square_results = {}

    for feature in features:
        plt.figure(figsize=(10, 5))

        # Histogram of real data
        sns.histplot(val_df[feature], kde=True, color='blue', label='Real Data')

        # Histogram of synthetic data
        sns.histplot(synthetic_dataset[feature], kde=True, color='orange', label='Generated Data')

        plt.xlabel(feature, fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend()
        plt.savefig(f'figures/{feature}_{count}_validation.pdf')

        # Compute Chi-square distance
        chi_sq = chi_square_distance(val_df[feature].dropna(), synthetic_dataset[feature].dropna())
        chi_square_results[feature] = chi_sq
        print(f"Chi-square distance for {feature}: {chi_sq:.4f}")

    results_df = pd.DataFrame({
        "Feature": features,
        "Chi-Square Distance": [chi_square_distance(val_df[f].dropna(), synthetic_dataset[f].dropna()) for f in
                                features]
    })
    results_df.to_csv(f"{count}_chi_square_results.csv", index=False)