import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
df = pd.read_csv("../data/cluster2GM_final.csv", parse_dates=['plug_in_datetime', 'plug_out_datetime'])
val_data = pd.read_csv("../data/val_ev_data_prepro.csv", parse_dates=['plug_in_datetime', 'plug_out_datetime'])
user_clusters_df = pd.read_csv('../data/user_clusters.csv')

# Merge cluster info into validation data
user_cluster_map = dict(zip(user_clusters_df['user_id'], user_clusters_df['user_cluster']))
val_data['user_cluster'] = val_data['user_id'].map(user_cluster_map)

# Feature and target definitions
features = ['plug_in_time', 'day_type', 'arrival_SoC', 'place']
target = 'next_CBS'
numeric_features = ['plug_in_time', 'arrival_SoC']
categorical_features = ['day_type', 'place']


# Define model
regressor = GradientBoostingRegressor(n_estimators=100)

# Group training data by user cluster
cluster_groups = df.groupby('user_cluster')

for cluster_id, cluster_df in cluster_groups:

    cluster_df = cluster_df.dropna(subset=[target] + features)
    X_train = cluster_df[features]
    y_train = cluster_df[target]

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    model_path = f"../models/GradientBoosting_next_CBS_cluster_{cluster_id}.pkl"
    joblib.dump(pipeline, model_path)

