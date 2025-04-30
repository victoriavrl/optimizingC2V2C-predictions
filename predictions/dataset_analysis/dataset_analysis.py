import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")

# Your default matplotlib settings
mpl.rcParams.update({'font.size': 16})
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Charter', 'XCharter', 'Georgia', 'Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'

df = pd.read_csv("../data/combined_charging_sessions.csv",
                 parse_dates=["plug_in_datetime", "plug_out_datetime"])

# Ensure datetime is parsed properly
df['month'] = df['plug_in_datetime'].dt.month
df['weekday'] = df['plug_in_datetime'].dt.day_name()
df['hour'] = df['plug_in_datetime'].dt.hour


def plot_workplace_weekday_frequency(df: pd.DataFrame):
    df['weekday'] = df['plug_in_datetime'].dt.weekday
    workplace_weekly = df[df['place'] == 'workplace']['weekday'].value_counts().sort_index()
    weekday_order = [0, 1, 2, 3, 4, 5, 6]
    workplace_weekly = workplace_weekly[weekday_order]

    weekday_map = {
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    }
    workplace_weekly.index = [weekday_map[x] for x in workplace_weekly.index]

    # Plot
    plt.figure(figsize=(8, 5))
    workplace_weekly.plot(kind='barh', color="#69A2B0")
    plt.ylabel("Day of Week")
    plt.xlabel("Number of Workplace Sessions")
    plt.tight_layout()
    plt.savefig("figures/workplace_weekday_freq.pdf")
    plt.close()


def plot_workplace_monthly_frequency(df: pd.DataFrame):
    df['month'] = df['plug_in_datetime'].dt.month

    workplace_monthly = df[df['place'] == 'workplace']['month'].value_counts().sort_index()

    all_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    workplace_monthly = workplace_monthly.reindex(all_months, fill_value=0)

    workplace_monthly = workplace_monthly[::-1]

    month_map = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
        7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }

    workplace_monthly.index = [month_map[x] for x in workplace_monthly.index]

    # Plot
    plt.figure(figsize=(8, 5))
    workplace_monthly.plot(kind='barh', color="#69A2B0")
    plt.ylabel("Month")
    plt.xlabel("Number of Workplace Sessions")
    plt.tight_layout()
    plt.savefig("figures/workplace_monthly_frequency.pdf")
    plt.close()


def plot_median_session_duration_per_place(df):
    avg_duration = df.groupby('place')['connected_duration'].median()

    # Plot
    plt.figure(figsize=(8, 6))
    avg_duration.plot(kind='barh', color="#69A2B0")
    plt.xlabel("Median Session Duration (hours)")
    plt.ylabel("Place")
    plt.tight_layout()
    plt.savefig("figures/median_session_duration_per_place.pdf")
    plt.close()


def plot_workplace_daily_median_duration(df: pd.DataFrame):
    df['weekday'] = df['plug_in_datetime'].dt.weekday

    workplace_weekday_median = df[df['place'] == 'workplace'].groupby('weekday')['connected_duration'].median()

    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    workplace_weekday_median = workplace_weekday_median[range(7)]
    workplace_weekday_median.index = weekday_names

    # Plot
    plt.figure(figsize=(10, 6))
    workplace_weekday_median.plot(kind='bar', color="#69A2B0")
    plt.xlabel("Day of Week")
    plt.xticks(rotation=40)
    plt.ylabel("Median Connected Duration (hours)")
    plt.tight_layout()
    plt.savefig("figures/workplace_weekday_median_duration.pdf")
    plt.close()


def plot_home_weekday_median_duration(df: pd.DataFrame):
    df['weekday'] = df['plug_in_datetime'].dt.weekday

    home_weekday_median = df[df['place'] == 'home'].groupby('weekday')['connected_duration'].median()

    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    home_weekday_median = home_weekday_median[range(7)]
    home_weekday_median.index = weekday_names

    # Plot
    plt.figure(figsize=(10, 6))
    home_weekday_median.plot(kind='bar', color="#69A2B0")
    plt.xlabel("Day of Week")
    plt.xticks(rotation=40)
    plt.ylabel("Median Connected Duration (hours)")
    plt.tight_layout()
    plt.savefig("figures/home_weekday_median_duration.pdf")
    plt.close()


def plot_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot the distribution of HBS
    axes[0, 0].hist(df['HBS'], bins=30, color="#69A2B0", edgecolor='black')
    axes[0, 0].set_title("Distribution of HBS")
    axes[0, 0].set_xlabel("HBS")
    axes[0, 0].set_ylabel("Frequency")

    # Plot the distribution of DBS
    axes[0, 1].hist(df['DBS'], bins=30, color="#659157", edgecolor='black')
    axes[0, 1].set_title("Distribution of DBS")
    axes[0, 1].set_xlabel("DBS")
    axes[0, 1].set_ylabel("Frequency")

    # Plot the distribution of plug_in_time
    axes[1, 0].hist(df['plug_in_datetime'].dt.hour, bins=24, color="#E05263", edgecolor='black')  # Hourly distribution
    axes[1, 0].set_title("Distribution of Plug-In Time")
    axes[1, 0].set_xlabel("Hour of Day")
    axes[1, 0].set_ylabel("Frequency")

    # Plot the distribution of connected_duration
    axes[1, 1].hist(df['connected_duration'], bins=30, color="#FFCAB1", edgecolor='black')
    axes[1, 1].set_title("Distribution of Connected Duration")
    axes[1, 1].set_xlabel("Connected Duration (hours)")
    axes[1, 1].set_ylabel("Frequency")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig("figures/distributions.pdf")
    plt.close()

def plot_median_dbs_by_weekday(df: pd.DataFrame):
    df['weekday'] = df['plug_in_datetime'].dt.weekday

    median_dbs = df.groupby('weekday')['DBS'].median()

    weekday_order = [0, 1, 2, 3, 4, 5, 6]
    median_dbs = median_dbs.reindex(weekday_order)

    weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    median_dbs.index = weekday_labels

    # Plot
    plt.figure(figsize=(8, 5))
    median_dbs.plot(kind='barh', color="#69A2B0", edgecolor='black')
    plt.xlabel("Median DBS")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig("figures/median_dbs_by_weekday.pdf")
    plt.close()

def plot_long_connection_frequency_by_weekday(df: pd.DataFrame):
    if 'connected_duration' not in df.columns:
        df['connected_duration'] = (df['plug_out_datetime'] - df['plug_in_datetime']).dt.total_seconds() / 3600

    long_sessions = df[df['connected_duration'] > 24].copy()

    long_sessions['weekday'] = long_sessions['plug_in_datetime'].dt.weekday

    weekday_counts = long_sessions['weekday'].value_counts().reindex(range(7), fill_value=0)

    weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts.index = weekday_labels

    # Plot
    plt.figure(figsize=(8, 5))
    weekday_counts.plot(kind='bar', color="#69A2B0", edgecolor='black')
    plt.xlabel("Day of Week")
    plt.ylabel("Frequency of Connections > 24h")
    plt.tight_layout()
    plt.savefig("figures/long_connection_duration_by_weekday.pdf")
    plt.close()


# Simple analysis
print(df.info())

# Run all plots
plot_workplace_weekday_frequency(df)
plot_workplace_monthly_frequency(df)
plot_median_session_duration_per_place(df)
plot_median_session_duration_per_place(df)
plot_workplace_daily_median_duration(df)
plot_home_weekday_median_duration(df)
plot_distributions(df)
plot_median_dbs_by_weekday(df)
plot_long_connection_frequency_by_weekday(df)

