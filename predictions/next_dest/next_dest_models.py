import pandas as pd
import os
import csv
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


file_path = "../data/cluster2GM_final.csv"
df = pd.read_csv(file_path, parse_dates=['plug_in_datetime', 'plug_out_datetime'])
val_data = pd.read_csv("../data/val_ev_data_prepro.csv", parse_dates=['plug_in_datetime', 'plug_out_datetime'])

user_clusters_df = pd.read_csv('../data/user_clusters.csv')
user_cluster_map = dict(zip(user_clusters_df['user_id'], user_clusters_df['user_cluster']))

val_data['user_cluster'] = val_data['user_id'].map(user_cluster_map)


features = ['plug_in_time', 'day_type', 'arrival_SoC', 'place']
target = 'next_dest'


clustered_data = df.groupby('user_cluster')
output_csv = "../data/random_search_results.csv"


if not os.path.exists(output_csv):
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_cluster', 'model', 'params', 'metric', 'score'])


for cluster_id, group in clustered_data:
    print(f"\n===== User Cluster {cluster_id} =====")

    val_cluster = val_data[val_data['user_cluster'] == cluster_id]
    if val_cluster.empty:
        print("⚠️ No validation data for this cluster. Skipping.")
        continue

    X_train = group[features]
    X_val = val_cluster[features]

    # Encode target labels
    le = LabelEncoder()
    y_train = le.fit_transform(group[target])
    y_val = le.transform(val_cluster[target])

    xgb_model = XGBClassifier(
        objective='multi:softmax',
        num_class=5,
        base_score=0.2,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), ['plug_in_time', 'arrival_SoC']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['day_type', 'place'])
        ])),
        ('classifier', xgb_model)
    ])

    param_grid = {
        'classifier__max_depth': [3, 5, 7, 9],
        'classifier__min_child_weight': [1, 5, 10],
        'classifier__gamma': [0, 0.1, 0.5, 1],
        'classifier__subsample': [0.6, 0.8, 1],
        'classifier__colsample_bytree': [0.6, 0.8, 1],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__n_estimators': [300, 500, 800],
        'classifier__reg_alpha': [0, 0.1, 1],
        'classifier__reg_lambda': [1, 5, 10],
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=50,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1,
        random_state=42,
        refit=True
    )

    random_search.fit(X_train, y_train)
    results_df = pd.DataFrame(random_search.cv_results_)

    for i in range(len(results_df)):
        params = results_df.loc[i, 'params']
        mean_score = results_df.loc[i, 'mean_test_score']

        with open(output_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                cluster_id,
                "XGBoost",
                params,
                'accuracy',
                mean_score
            ])
    best_model = random_search.best_estimator_

    y_pred_val = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)

    print(f"Best Parameters for Cluster {cluster_id}: {random_search.best_params_}")
    print(f"Validation Accuracy: {accuracy:.3f}")

