import os
import csv

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv("data/cluster2GM_final.csv", parse_dates=['plug_in_datetime', 'plug_out_datetime'])
val_data = pd.read_csv("data/val_ev_data_prepro.csv", parse_dates=['plug_in_datetime', 'plug_out_datetime'])
user_clusters_df = pd.read_csv('data/user_clusters.csv')

user_cluster_map = dict(zip(user_clusters_df['user_id'], user_clusters_df['user_cluster']))
val_data['user_cluster'] = val_data['user_id'].map(user_cluster_map)


features = ['plug_in_time', 'day_type', 'arrival_SoC', 'place']
target = 'next_CBS'
numeric_features = ['plug_in_time', 'arrival_SoC']
categorical_features = ['day_type', 'place']


os.makedirs("models", exist_ok=True)
results_csv_path = "data/next_CBS_prediction_results.csv"


with open(results_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'method', 'user_cluster', 'plug_in_datetime', 'place', 'arrival_SoC',
                     'predicted_next_CBS', 'actual_next_CBS', 'error'])

models_to_test = {
    "RandomForest": RandomForestRegressor(n_estimators=100),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=20),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100),
    "MLPRegressor": MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500
    )
}

# Group training data by user cluster
cluster_groups = df.groupby('user_cluster')

for model_name, regressor in models_to_test.items():
    print(f"\n===== Testing Model: {model_name} =====")

    for cluster_id, cluster_df in cluster_groups:
        print(f"\nTraining {model_name} for cluster {cluster_id}")

        cluster_df = cluster_df.dropna(subset=[target] + features)
        X_train = cluster_df[features]
        y_train = cluster_df[target]

        # Preprocessing
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

        # Full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Save model
        model_path = f"models/{model_name}_cluster_{cluster_id}.pkl"
        joblib.dump(pipeline, model_path)

        # Evaluate
        val_cluster = val_data[val_data['user_cluster'] == cluster_id].dropna(subset=features + [target])
        if not val_cluster.empty:
            X_val = val_cluster[features]
            y_val = val_cluster[target]
            y_pred = pipeline.predict(X_val)

            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            print(f"{model_name} | Cluster {cluster_id} — MAE: {mae:.2f}, RMSE: {rmse:.2f}")

            with open(results_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                for i, (idx, row) in enumerate(val_cluster.iterrows()):
                    pred = y_pred[i]
                    actual = row['next_CBS']
                    error = abs(pred - actual)
                    writer.writerow([
                        row['user_id'],
                        model_name,
                        cluster_id,
                        row['plug_in_datetime'],
                        row['place'],
                        row['arrival_SoC'],
                        pred,
                        actual,
                        error
                    ])
        else:
            print(f"No validation data for cluster {cluster_id}.")
