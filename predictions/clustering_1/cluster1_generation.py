import seaborn as sns
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

final_dataset = pd.read_csv("../data/final_clustered_ev_drivers.csv")
train_df = pd.read_csv('data/cluster1_train.csv', parse_dates=['plug_in_datetime', 'plug_out_datetime'])
charging_behavior = pd.read_csv('data/charging_behavior.csv')
val_df = pd.read_csv('data/cluster1_val.csv', parse_dates=['plug_in_datetime', 'plug_out_datetime'])

original_dataset = train_df.copy()
merged_dataset = original_dataset.merge(final_dataset, on='user_id', how='left')
merged_dataset.info()

merged_dataset['weekday'] = merged_dataset['plug_in_datetime'].dt.weekday
merged_dataset['is_weekend'] = merged_dataset['weekday'].apply(lambda x: 1 if x >= 5 else 0)

clusters = merged_dataset['cluster_step1'].unique()
train_df = merged_dataset.copy()

train_df = train_df.merge(charging_behavior[['user_id', 'periodDays']], on='user_id', how='left')
timestamp = (val_df['plug_out_datetime'].max() - val_df['plug_in_datetime'].min()).total_seconds() / 3600


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
    freq_values = cluster_data['freq'].dropna().values * timestamp

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


# Generate synthetic charging sessions based on training data
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


# Validation

# Function to compute Chi-square histogram distance
def chi_square_distance(real_data, synthetic_data, bins=50):
    real_hist, bin_edges = np.histogram(real_data, bins=bins, density=True)
    synthetic_hist, _ = np.histogram(synthetic_data, bins=bin_edges, density=True)

    chi_sq = 0.5 * np.sum((real_hist - synthetic_hist) ** 2 / (real_hist + synthetic_hist + 1e-10))
    return chi_sq


# Plot histograms and compute Chi-square distance
features = ["plug_in_time", "connected_duration", "total_energy_grid"]
chi_square_results = {}

for feature in features:
    plt.figure(figsize=(10, 5))

    # Histogram of real data
    sns.histplot(val_df[feature], kde=True, color='blue', label='Real Data')

    # Histogram of synthetic data
    sns.histplot(synthetic_dataset[feature], kde=True, color='orange', label='Generated Data')

    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.savefig(f'figures/{feature}_validation.pdf')


    # Compute Chi-square distance
    chi_sq = chi_square_distance(val_df[feature].dropna(), synthetic_dataset[feature].dropna())
    chi_square_results[feature] = chi_sq
    print(f"Chi-square distance for {feature}: {chi_sq:.4f}")

# Print final results
print("Chi-square Distance Summary:")
for feature, chi_sq in chi_square_results.items():
    print(f"{feature}: {chi_sq:.4f}")

