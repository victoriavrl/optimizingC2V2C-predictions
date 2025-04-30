import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier


df = pd.read_csv("../data/cluster2GM_final.csv", parse_dates=['plug_in_datetime', 'plug_out_datetime'])

features = ['plug_in_time', 'day_type', 'arrival_SoC', 'place']
target = 'next_dest'
numeric_features = ['plug_in_time', 'arrival_SoC']
categorical_features = ['day_type', 'place']

# Train a separate model for each user cluster
for cluster in df['user_cluster'].unique():
    cluster_df = df[df['user_cluster'] == cluster].copy()

    X = cluster_df[features]
    y_raw = cluster_df[target]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    num_classes = len(le.classes_)

    with open(f"../models/xgb_next_dest_encoder_C{cluster}.pkl", "wb") as f:
        pickle.dump(le, f)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    model = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        learning_rate=0.1,
        max_depth=20,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1,
        eval_metric='mlogloss'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X, y)

    model_path = f"../models/xgb_next_dest_model_C{cluster}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
