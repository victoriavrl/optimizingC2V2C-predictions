import pandas as pd
from scipy.stats import gaussian_kde
import os

file_path = "../data/cluster2GM_final.csv"
df = pd.read_csv(file_path, parse_dates=['plug_in_datetime', 'plug_out_datetime'])
user_clusters_df = pd.read_csv('../data/user_clusters.csv')


def predict_plug_out_median(user_cluster, datetime, place):
    user_type = df[df['user_cluster'] == user_cluster]
    close_sessions = user_type[user_type['place'] == place]

    datetime = pd.to_datetime(datetime)
    time = datetime.hour
    if place == "home":
        day = 'Weekday' if datetime.dayofweek < 4 else (
                'Friday' if datetime.dayofweek == 4 else (
                    'Saturday' if datetime.dayofweek == 5 else 'Sunday'
                )
            )
        close_sessions = close_sessions[close_sessions["day_type"] == day]

    close_sessions = close_sessions[
        (close_sessions['plug_in_time'] >= time - 1) &
        (close_sessions['plug_in_time'] <= time + 1)
        ]

    if close_sessions.empty:
        print(f"No close sessions found for {user_cluster}, {datetime}, {place}")
        return None, None

    most_frequent_session_cluster = close_sessions['SessionCluster'].mode()[0]

    session_type = close_sessions[close_sessions['SessionCluster'] == most_frequent_session_cluster]

    connected_median = session_type['connected_duration'].median()

    plug_out_time_exp = datetime + pd.Timedelta(hours=connected_median).round('s')

    plug_out_time_exp = plug_out_time_exp.round('S')

    return plug_out_time_exp


# Prediction using KDE

def get_kde_sample(data, fallback=None):
    """Returns a sampled value from KDE if enough data is present, otherwise returns a fallback."""
    try:
        if len(data) > 1:
            kde = gaussian_kde(data)
            return kde.resample(1)[0].item()
        elif len(data) == 1:
            return data.iloc[0]
        else:
            return fallback
    except Exception as e:
        print(f"Error occurred during KDE sampling: {e}")
        return fallback


def predict_plug_out_kde(user_cluster, datetime, place):
    user_type = df[df['user_cluster'] == user_cluster]

    close_sessions = user_type[user_type['place'] == place]

    datetime = pd.to_datetime(datetime)
    time = datetime.hour
    if place == "home":
        day = 'Weekday' if datetime.dayofweek < 4 else (
                'Friday' if datetime.dayofweek == 4 else (
                    'Saturday' if datetime.dayofweek == 5 else 'Sunday'
                )
            )
        close_sessions = close_sessions[close_sessions["day_type"] == day]

    close_sessions = close_sessions[
        (close_sessions['plug_in_time'] >= time - 1) &
        (close_sessions['plug_in_time'] <= time + 1)
        ]

    if close_sessions.empty:
        print(f"No close sessions found for {user_cluster}, {datetime}, {place}")
        return None, None

    most_frequent_session_cluster = close_sessions['SessionCluster'].mode()[0]

    session_type = close_sessions[close_sessions['SessionCluster'] == most_frequent_session_cluster]

    connected_duration_sample = get_kde_sample(session_type['connected_duration'],
                                               fallback=session_type['connected_duration'].median())

    plug_out_time_exp = datetime + pd.Timedelta(hours=connected_duration_sample).round('s')

    return plug_out_time_exp


def evaluate_prediction_methods(ev_data, methods, output_file="../data/plug_out_prediction_results_days.csv"):
    input_cols = ['plug_in_datetime', 'place', 'arrival_SoC']
    output_cols = ['plug_out_datetime', 'next_CBS', 'next_dest']
    user_cluster_map = dict(zip(user_clusters_df['user_id'], user_clusters_df['user_cluster']))

    results = []

    # If the file exists, remove it so we can write fresh headers
    if os.path.exists(output_file):
        os.remove(output_file)

    for index, session in ev_data.iterrows():
        user_id = session['user_id']
        inputs = session[input_cols]
        actual_outputs = session[output_cols]
        user_cluster = user_cluster_map[user_id]

        for method_name, method in methods.items():
            predicted_plug_out = method(
                user_cluster, inputs['plug_in_datetime'], inputs['place']
            )

            time_error = (predicted_plug_out - actual_outputs['plug_out_datetime']).total_seconds() / 3600 \
                if predicted_plug_out else None

            row = {
                "user_id": user_id,
                "user_cluster": user_cluster,
                "method_name": f"{method_name}_days",
                "plug_in_datetime": inputs['plug_in_datetime'],
                "place": inputs['place'],
                "actual_plug_out_datetime": actual_outputs['plug_out_datetime'],
                "predicted_plug_out_datetime": predicted_plug_out,
                "Time Error (hours)": time_error,
            }

            results.append(row)

            pd.DataFrame([row]).to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

    return pd.DataFrame(results)


val = pd.read_csv('../data/val_ev_data_prepro.csv', parse_dates=['plug_in_datetime', 'plug_out_datetime'])
val = val[val['arrival_SoC'] > 0]
val = val[val['connected_duration'] < 48]
methods = {
    "Median-Based": predict_plug_out_median,
    "KDE-Based": predict_plug_out_kde
}
evaluate_prediction_methods(val, methods)
