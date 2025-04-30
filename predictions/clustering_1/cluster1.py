import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.model_selection import train_test_split


df = pd.read_csv("../data/combined_charging_sessions.csv", parse_dates=['plug_in_datetime', 'plug_out_datetime'])
print(df.info())
print("Number of users", df["user_id"].nunique())

# Dataset cleaning
# Filter based on energy session
df = df[df["total_energy_grid"] <= 120]
# Filter based on connection time
df = df[df["connected_duration"] <= 50]

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
train_df.to_csv("data/cluster1_train.csv", index=False)
val_df.to_csv("data/cluster1_val.csv", index=False)
# Data preprocessing

# Theoretical sessions based on a mean session per user
theoretical_sessions = train_df.groupby("user_id").agg(
    mean_plugin_time=("plug_in_time", "mean"),
    mean_connection_time=("connected_duration", "mean"),
    mean_energy_session=("total_energy_grid", "mean")
).reset_index()

# Frequency of charging
# Compute frequency of charging: sessions per day
last_day = train_df["plug_out_datetime"].max()

charging_behavior = train_df.groupby("user_id").agg(
    numSessions=("plug_in_datetime", "count"),
    firstSession=("plug_in_datetime", "min")
).reset_index()


charging_behavior["firstSession"] = pd.to_datetime(charging_behavior["firstSession"])
charging_behavior["periodDays"] = (last_day - charging_behavior["firstSession"]).dt.days + 1

charging_behavior["freq"] = charging_behavior["numSessions"] / charging_behavior["periodDays"]

final_dataset = pd.merge(theoretical_sessions, charging_behavior[["user_id", "freq"]], on="user_id", how="inner")

charging_behavior.to_csv("data/charging_behavior.csv", index=False)

# Clustering Step 1

features_step1 = ["mean_plugin_time", "mean_connection_time", "mean_energy_session"]
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

print(f"Optimal number of clusters: {optimal_k1}")

# Plot the Elbow curve
plt.figure(figsize=(6, 4))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.axvline(x=optimal_k1, color='r', linestyle='--', label=f"Elbow at k={optimal_k1}")
plt.xlabel("Number of Clusters (k)",fontsize=12)
plt.ylabel("Inertia (Sum of Squared Distances)",fontsize=12)
plt.savefig("figures/elbow_curve.pdf")

kmeans_step1 = KMeans(n_clusters=optimal_k1, random_state=42, n_init=10)
final_dataset["cluster_step1"] = kmeans_step1.fit_predict(df_scaled_step1)

# Clustering Step 2

subclusters = {}

for cluster in final_dataset["cluster_step1"].unique():
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
            final_dataset.loc[
                final_dataset["cluster_step1"] == cluster, "subcluster"] = -1  # Indicating no sub-clustering

# Save final clustered dataset
final_dataset.to_csv("data/final_clustered_ev_drivers.csv", index=False)

# Analysis of the clusters

summary = final_dataset.groupby("cluster_step1")[
    ["mean_plugin_time", "mean_connection_time", "mean_energy_session", "freq"]
].mean().reset_index()

# Create a summary table
summary_table = []

for cluster in summary["cluster_step1"].unique():
    cluster_data = summary[summary["cluster_step1"] == cluster].iloc[0]

    # Times are already in hours, so we format them directly
    plugin_time_str = f"{int(cluster_data['mean_plugin_time'])}h{int((cluster_data['mean_plugin_time'] % 1) * 60):02d}"
    parking_time_str = f"{int(cluster_data['mean_connection_time'])}h{int((cluster_data['mean_connection_time'] % 1) * 60):02d}"

    # Mean energy usage
    energy_usage = round(cluster_data["mean_energy_session"], 2)

    # Create the row for this cluster
    summary_table.append([
        f"Cluster {cluster}",
        round(cluster_data["freq"], 3),  # Frequency of sessions, rounded for clarity
        len(final_dataset[final_dataset["cluster_step1"] == cluster]["user_id"].unique()),  # Number of drivers
        plugin_time_str,  # Mean plug-in time in "XhYY" format
        parking_time_str,  # Mean parking time in "XhYY" format
        energy_usage,  # Mean energy usage in kWh
    ])

# Convert to DataFrame
summary_df = pd.DataFrame(summary_table, columns=[
    "Cluster ID", "Freq of Sessions", "# of Drivers",
    "Plug-In Time (Mean Value)", "Parking Time (Mean Value)", "Energy (Mean [kWh])"
])

summary_df.to_csv('figures/summary_clusters.txt')

# Get the summary and reset index for easier manipulation
summary = final_dataset.groupby(["cluster_step1", "subcluster"])[
    ["mean_plugin_time", "mean_connection_time", "mean_energy_session", "freq"]
].mean().reset_index()

# Create a summary table
summary_table = []

for cluster in summary["cluster_step1"].unique():
    for subcluster in summary[summary["cluster_step1"] == cluster]["subcluster"].unique():
        cluster_data = summary[(summary["cluster_step1"] == cluster) & (summary["subcluster"] == subcluster)].iloc[0]

        # Convert times from minutes to hours and minutes
        plugin_time = cluster_data["mean_plugin_time"]
        parking_time = cluster_data["mean_connection_time"]

        # Times are already in hours, so we format them directly
        plugin_time_str = f"{int(cluster_data['mean_plugin_time'])}h{int((cluster_data['mean_plugin_time'] % 1) * 60):02d}"
        parking_time_str = f"{int(cluster_data['mean_connection_time'])}h{int((cluster_data['mean_connection_time'] % 1) * 60):02d}"

        # Mean energy usage
        energy_usage = round(cluster_data["mean_energy_session"], 2)

        # Count number of drivers in the sub-cluster
        num_drivers = len(final_dataset[
                              (final_dataset["cluster_step1"] == cluster) &
                              (final_dataset["subcluster"] == subcluster)
                              ]["user_id"].unique())

        # Create the row for this cluster and sub-cluster
        summary_table.append([
            f"Cluster {cluster}",
            round(cluster_data["freq"], 3),  # Frequency of sessions, rounded
            num_drivers,  # Number of drivers in the sub-cluster
            plugin_time_str,  # Mean plug-in time in "XhYY" format
            parking_time_str,  # Mean parking time in "XhYY" format
            energy_usage,  # Mean energy usage in kWh
            subcluster  # Sub-cluster ID
        ])

# Convert to DataFrame
summary_df = pd.DataFrame(summary_table, columns=[
    "Cluster ID", "Freq of Sessions", "# of Drivers",
    "Plug-In Time (Mean Value)", "Parking Time (Mean Value)", "Energy (Mean [kWh])", "Sub-Clusters"
])

# Display the summary table
df.to_csv('data/summary_sub_clusters.csv')



