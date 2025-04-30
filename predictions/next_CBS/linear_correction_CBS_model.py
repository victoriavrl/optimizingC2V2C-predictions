import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import numpy as np

pred = pd.read_csv('data/next_CBS_prediction_results.csv', parse_dates=['plug_in_datetime'])

gradient_boost_data = pred[pred['method'] == 'GradientBoosting']
X_gradient_boost = gradient_boost_data['predicted_next_CBS'].values.reshape(-1, 1)
y_gradient_boost = gradient_boost_data['actual_next_CBS'].values

X_train_gradient_boost, X_val_gradient_boost, y_train_gradient_boost, y_val_gradient_boost = train_test_split(
    X_gradient_boost, y_gradient_boost, test_size=0.2, random_state=42)

correction_model_gradient_boost = LinearRegression()
correction_model_gradient_boost.fit(X_train_gradient_boost, y_train_gradient_boost)

y_val_gradient_boost_pred_corrected = correction_model_gradient_boost.predict(X_val_gradient_boost)
# Compute RMSE for Gradient Boosting model
original_rmse_gradient_boost = np.sqrt(mean_squared_error(y_val_gradient_boost, X_val_gradient_boost.flatten()))  # RMSE for the predictions
corrected_rmse_gradient_boost = np.sqrt(mean_squared_error(y_val_gradient_boost, y_val_gradient_boost_pred_corrected))  # RMSE after correction
original_mae_gradient_boost = mean_absolute_error(y_val_gradient_boost, X_val_gradient_boost.flatten())  # MAE for the predictions
corrected_mae_gradient_boost = mean_absolute_error(y_val_gradient_boost, y_val_gradient_boost_pred_corrected)  # MAE after correction
print("Gradient Boosting Model Validation RMSE before correction:", original_rmse_gradient_boost, "kWh")
print("Gradient Boosting Model Validation RMSE after correction: ", corrected_rmse_gradient_boost, "kWh")
print("Gradient Boosting Model Validation MAE before correction:", original_mae_gradient_boost, "kWh")
print("Gradient Boosting Model Validation MAE after correction: ", corrected_mae_gradient_boost, "kWh")

complete_model = LinearRegression()
complete_model.fit(X_gradient_boost, y_gradient_boost)
joblib.dump(complete_model, 'models/linear_correction_gradient_boost_model.pkl')
