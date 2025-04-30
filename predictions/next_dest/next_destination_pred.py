import pickle
import pandas as pd
import os
import csv

file_path = "../data/cluster2GM_final.csv"
df = pd.read_csv(file_path, parse_dates=['plug_in_datetime', 'plug_out_datetime'])

val_data = pd.read_csv("../data/val_ev_data_prepro.csv", parse_dates=['plug_in_datetime', 'plug_out_datetime'])

output_file = "../data/next_dest_prediction_results_15.csv"


def predict_nd_mode(user_cluster, datetime, place, SoC):
    user_type = df[df['user_cluster'] == user_cluster]
    close_sessions = user_type[user_type['place'] == place]

    datetime = pd.to_datetime(datetime)
    time = datetime.hour

    close_sessions = close_sessions[
        (close_sessions['plug_in_time'] >= time - 0.25) &
        (close_sessions['plug_in_time'] <= time + 0.25)
        ]

    if close_sessions.empty:
        print(f"No close sessions found for {user_cluster}, {datetime}, {place}")
        return None

    most_frequent_session_cluster = close_sessions['SessionCluster'].mode()[0]
    session_type = close_sessions[close_sessions['SessionCluster'] == most_frequent_session_cluster]

    if not session_type['next_dest'].empty:
        next_destination = session_type['next_dest'].mode()[0]
    else:
        next_destination = None

    return next_destination


def predict_nd_xgb(user_cluster, datetime, place, SoC):
    with open(f"../models/xgb_next_dest_model_C{user_cluster}.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"../models/xgb_next_dest_encoder_C{user_cluster}.pkl", "rb") as f:
        encoder = pickle.load(f)

    datetime = pd.to_datetime(datetime)
    time = datetime.hour
    inputs = pd.DataFrame({
        'plug_in_time': [time],
        'day_type': ['Saturday' if datetime.dayofweek == 5 else ('Sunday' if datetime.dayofweek == 6 else 'Weekday')],
        'place': [place],
        'arrival_SoC': [SoC]
    })

    try:
        encoded_pred = model.predict(inputs)[0]
        nd = encoder.inverse_transform([encoded_pred])[0]
    except Exception as e:
        print(f"XGB prediction error: {e}")
        nd = None

    return nd



if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'user_id', 'method', 'user_cluster', 'plug_in_datetime', 'place', 'arrival_SoC',
            'predicted_next_dest', 'actual_next_dest', 'correct', 'missing'
        ])

user_clusters_df = pd.read_csv('data/user_clusters.csv')
correct = 0
total = 0
missing = 0
user_cluster_map = dict(zip(user_clusters_df['user_id'], user_clusters_df['user_cluster']))

for _, row in val_data.iterrows():
    actual = row['next_dest']
    user = row['user_id']
    plug_time = row['plug_in_datetime']
    place = row['place']
    soc = row['arrival_SoC']
    user_cluster = user_cluster_map[user]

    # --- Method 1: mode-based ---
    pred_mode = predict_nd_mode(user_cluster, plug_time, place, soc)
    is_correct_mode = pred_mode == actual if pred_mode is not None else False
    is_missing_mode = pred_mode is None

    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            user, "mode_15", user_cluster, plug_time, place, soc,
            pred_mode, actual, is_correct_mode, is_missing_mode
        ])

    # --- Method 2: XGBoost-based ---
    pred_xgb = predict_nd_xgb(user_cluster, plug_time, place, soc)
    is_correct_xgb = pred_xgb == actual if pred_xgb is not None else False
    is_missing_xgb = pred_xgb is None

    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)
    writer.writerow([
            user, "xgb", user_cluster, plug_time, place, soc,
            pred_xgb, actual, is_correct_xgb, is_missing_xgb
        ])
