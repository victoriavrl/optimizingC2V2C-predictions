import pandas as pd
import joblib

df = pd.read_csv("data/cluster2GM_final.csv")


def predict_ev_charging(user_cluster, datetime, place, SoC):
    # Filter the data by user_cluster
    user_type = df[df['user_cluster'] == user_cluster]

    # Further filter by place
    close_sessions = user_type[user_type['place'] == place]

    # Convert the input datetime and extract the hour
    datetime = pd.to_datetime(datetime)
    time = datetime.hour

    # Filter close sessions based on the plug-in time
    close_sessions = close_sessions[
        (close_sessions['plug_in_time'] >= time - 0.5) &
        (close_sessions['plug_in_time'] <= time + 0.5)
        ]

    if close_sessions.empty:
        print(f"No close sessions found for {user_cluster}, {datetime}, {place}")
        return None, None

    # Determine the most frequent session cluster in close_sessions
    most_frequent_session_cluster = close_sessions['SessionCluster'].mode()[0]

    # Filter the close_sessions based on the most frequent session cluster
    session_type = close_sessions[close_sessions['SessionCluster'] == most_frequent_session_cluster]

    # NEXT PLUG-OUT TIME
    connected_median = session_type['connected_duration'].median()
    plug_out_time_exp = datetime + pd.Timedelta(hours=connected_median).round('s')

    plug_out_time_exp = plug_out_time_exp.round('min')

    # NEXT DESTINATION
    file_path = f"../predictions/models/xgb_next_dest_model_C{user_cluster}.pkl"
    model_nd = joblib.load(file_path)
    label_encoder = joblib.load(f"../predictions/models/xgb_next_dest_encoder_C{user_cluster}.pkl")
    inputs = pd.DataFrame({
        'plug_in_time': [time],
        'day_type': ['Saturday' if datetime.dayofweek == 5 else ('Sunday' if datetime.dayofweek == 6 else 'Weekday')],
        'place': [place],
        'arrival_SoC': [SoC]
    })
    next_destination = model_nd.predict(inputs)
    next_destination = label_encoder.inverse_transform(next_destination)[0]

    # NEXT CBS
    model_CBS = joblib.load(f"../predictions/models/GradientBoosting_next_CBS_cluster_{user_cluster}.pkl")
    energy_needed_exp = model_CBS.predict(inputs)[0]


    return plug_out_time_exp, energy_needed_exp, next_destination
