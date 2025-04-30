import pandas as pd
import numpy as np


def compute_iqr_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper


def preprocess_charging_data(df, next_cbs_bounds):
    df = df.drop(columns=['Unnamed: 0', 'prev_plug_out'], errors='ignore')

    # Keep only valid rows
    df = df[(df['arrival_SoC'] > 0) & (df['departure_SoC'] > 0) & (df['next_CBS'] > 0)]

    # Correct connected_duration and charging_duration
    df['connected_duration'] = df['connected_duration'].apply(lambda x: x % 48 if x > 48 else x)
    df['charging_duration'] = df['charging_duration'].apply(lambda x: x % 48 if x > 48 else x)

    # Correct HBS
    df['HBS'] = df['HBS'].apply(lambda x: x % 168 if x > 168 else x)

    df['plug_in_datetime'] = pd.to_datetime(df['plug_in_datetime'])
    df['plug_out_datetime'] = df['plug_in_datetime'] + pd.to_timedelta(df['connected_duration'], unit='h')
    df['plug_out_time'] = df['plug_out_datetime'].dt.hour + df['plug_out_datetime'].dt.minute / 60

    # Compute log_DBS
    df['log_DBS'] = np.log10(df['DBS'] + 1)

    # Filter based on provided IQR bounds for next_CBS
    lower_bound, upper_bound = next_cbs_bounds
    df = df[(df['next_CBS'] >= lower_bound) & (df['next_CBS'] <= upper_bound)]

    return df


# Training set
df = pd.read_csv('data/combined_charging_sessions.csv', parse_dates=['plug_in_datetime', 'plug_out_datetime'])
iqr_bounds = compute_iqr_bounds(df['next_CBS'])
df = preprocess_charging_data(df, iqr_bounds)
df.to_csv('data/combined_charging_sessions_prepro.csv', index=False)

# Validation set
val_df = pd.read_csv('data/val_ev_data.csv', parse_dates=['plug_in_datetime', 'plug_out_datetime'])
val_df = preprocess_charging_data(val_df, iqr_bounds)
val_df.to_csv('data/val_ev_data_prepro.csv', index=False)
