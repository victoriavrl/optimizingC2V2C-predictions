import pandas as pd
import numpy as np


files = {
    "plug_out_60": "data/plug_out_prediction_results_60.csv",
    "plug_out_15": "data/plug_out_prediction_results_15.csv",
    "plug_out_30": "data/plug_out_prediction_results_30.csv",
    "plug_out_days": "data/plug_out_prediction_results_days.csv",
    "next_dest": "data/next_dest_prediction_results.csv",
    "next_dest_15": "data/next_dest_prediction_results_15.csv",
    "next_dest_30": "data/next_dest_prediction_results_30.csv",
    "next_cbs": "data/next_CBS_prediction_results.csv"

}

results = []

# 1. Plug-out time (MAE, RMSE) for all plug_out files
for key in ["plug_out_15", "plug_out_30", "plug_out_60","plug_out_days"]:
    df = pd.read_csv(files[key])

    # Compute the time error in hours
    df['Time Error (hours)'] = (pd.to_datetime(df['predicted_plug_out_datetime']) - pd.to_datetime(
        df['actual_plug_out_datetime'])).dt.total_seconds() / 3600

    # Compute errors per method
    for method in df["method_name"].unique():
        method_df = df[df["method_name"] == method]
        errors = method_df["Time Error (hours)"].dropna()
        mae = errors.abs().mean()
        rmse = np.sqrt((errors ** 2).mean())
        results.append({
            "PredictionType": key,
            "Method": method,
            "MAE": mae,
            "RMSE": rmse
        })

# 2. Next destination (Success Rate for all next_dest files)
for key in ["next_dest", "next_dest_15", "next_dest_30"]:
    df = pd.read_csv(files[key])
    for method in df["method"].unique():
        method_df = df[df["method"] == method]
        correct_preds = method_df["correct"].dropna()
        success_rate = correct_preds.mean()
        results.append({
            "PredictionType": key,
            "Method": method,
            "SuccessRate": success_rate
        })

# 3. Next CBS (MAE, RMSE)
df = pd.read_csv(files["next_cbs"])
for method in df["method"].unique():
    method_df = df[df["method"] == method]
    errors = method_df["error"].dropna()
    mae = errors.abs().mean()
    rmse = np.sqrt((errors ** 2).mean())
    results.append({
        "PredictionType": "next_cbs",
        "Method": method,
        "MAE": mae,
        "RMSE": rmse
    })

# Format to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("data/evaluation_metrics.csv", index=False)
print("Evaluation complete. Results saved to evaluation_metrics.csv.")