import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import numpy as np


pred = pd.read_csv("data/plug_out_prediction_results.csv", parse_dates=["actual_plug_out_datetime",
                                                                         "predicted_plug_out_datetime"])
ref_time = min(pred["actual_plug_out_datetime"].min(), pred["predicted_plug_out_datetime"].min())

pred["actual_plug_out_numeric"] = (pred["actual_plug_out_datetime"] - ref_time).dt.total_seconds() / 60
pred["predicted_plug_out_numeric"] = (pred["predicted_plug_out_datetime"] - ref_time).dt.total_seconds() / 60

median_based_data = pred[pred["method_name"] == "Median-Based"]
kde_based_data = pred[pred["method_name"] == "KDE-Based"]

# === Median-Based model ===
X_median = median_based_data["predicted_plug_out_numeric"].values.reshape(-1, 1)
y_median = median_based_data["actual_plug_out_numeric"].values
X_train_median, X_val_median, y_train_median, y_val_median = train_test_split(X_median, y_median, test_size=0.2, random_state=42)

correction_model_median = LinearRegression()
correction_model_median.fit(X_train_median, y_train_median)

y_val_median_pred_corrected = correction_model_median.predict(X_val_median)

# Metrics for Median-Based
original_rmse_median = np.sqrt(mean_squared_error(y_val_median, X_val_median.flatten()))
corrected_rmse_median = np.sqrt(mean_squared_error(y_val_median, y_val_median_pred_corrected))
original_mae_median = mean_absolute_error(y_val_median, X_val_median.flatten())
corrected_mae_median = mean_absolute_error(y_val_median, y_val_median_pred_corrected)

print("Median-Based Model Validation RMSE before correction:", round(original_rmse_median, 2), "minutes")
print("Median-Based Model Validation RMSE after correction: ", round(corrected_rmse_median, 2), "minutes")
print("Median-Based Model Validation MAE before correction: ", round(original_mae_median, 2), "minutes")
print("Median-Based Model Validation MAE after correction:  ", round(corrected_mae_median, 2), "minutes")

# === KDE-Based model ===
X_kde = kde_based_data["predicted_plug_out_numeric"].values.reshape(-1, 1)
y_kde = kde_based_data["actual_plug_out_numeric"].values
X_train_kde, X_val_kde, y_train_kde, y_val_kde = train_test_split(X_kde, y_kde, test_size=0.2, random_state=42)

correction_model_kde = LinearRegression()
correction_model_kde.fit(X_train_kde, y_train_kde)

y_val_kde_pred_corrected = correction_model_kde.predict(X_val_kde)

# Metrics for KDE-Based
original_rmse_kde = np.sqrt(mean_squared_error(y_val_kde, X_val_kde))
corrected_rmse_kde = np.sqrt(mean_squared_error(y_val_kde, y_val_kde_pred_corrected))
original_mae_kde = mean_absolute_error(y_val_kde, X_val_kde)
corrected_mae_kde = mean_absolute_error(y_val_kde, y_val_kde_pred_corrected)

print("KDE-Based Model Validation RMSE before correction:", round(original_rmse_kde, 2), "minutes")
print("KDE-Based Model Validation RMSE after correction: ", round(corrected_rmse_kde, 2), "minutes")
print("KDE-Based Model Validation MAE before correction: ", round(original_mae_kde, 2), "minutes")
print("KDE-Based Model Validation MAE after correction:  ", round(corrected_mae_kde, 2), "minutes")

# Save models
joblib.dump(correction_model_median, 'models/correction_model_median.pkl')
joblib.dump(correction_model_kde, 'models/correction_model_kde.pkl')

pred["corrected_plug_out_numeric_median"] = correction_model_median.predict(pred["predicted_plug_out_numeric"].values.reshape(-1, 1))
pred["corrected_plug_out_numeric_kde"] = correction_model_kde.predict(pred["predicted_plug_out_numeric"].values.reshape(-1, 1))

pred["corrected_plug_out_datetime_median"] = pd.to_datetime(pred["corrected_plug_out_numeric_median"] * 60, unit='s', origin=ref_time)
pred["corrected_plug_out_datetime_kde"] = pd.to_datetime(pred["corrected_plug_out_numeric_kde"] * 60, unit='s', origin=ref_time)

pred["corrected_error_median_minutes"] = (pred["actual_plug_out_datetime"] - pred["corrected_plug_out_datetime_median"]).dt.total_seconds() / 60
pred["corrected_error_kde_minutes"] = (pred["actual_plug_out_datetime"] - pred["corrected_plug_out_datetime_kde"]).dt.total_seconds() / 60

pred.to_csv("data/plug_out_prediction_results_corrected.csv", index=False)
