"""
train_model.py
---------------
Full ML workflow for the Employee Salary Predictor:
  1. Load data
  2. Preprocess (handle missing values, encode categoricals, scale numerics)
  3. Train multiple regression models
  4. Evaluate with MAE, RMSE, R2
  5. Save the best-performing model + preprocessing objects for the Streamlit app
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("data/employee_salary.csv")
print(f"Loaded dataset with shape: {df.shape}")

# ---------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------
# Handle missing numeric values with median imputation
for col in ["Age", "Years of Experience"]:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

categorical_cols = ["Gender", "Education Level", "Department", "Job Title", "Location"]
numeric_cols = ["Age", "Years of Experience"]

# Label-encode categorical columns (and keep the encoders for the app)
encoders = {}
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le

feature_cols = numeric_cols + categorical_cols
X = df_encoded[feature_cols]
y = df_encoded["Salary"]

# Scale numeric features (tree models don't need it, but Linear Regression does)
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# ---------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# 4. Train multiple models & evaluate
# ---------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    trained_models[name] = model
    print(f"\n{name}")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  R2   : {r2:.4f}")

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n=== Model comparison (sorted by R2) ===")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------
# 5. Pick best model and save everything needed by the Streamlit app
# ---------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name}")

joblib.dump(best_model, "model/salary_model.pkl")
joblib.dump(encoders, "model/encoders.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(feature_cols, "model/feature_cols.pkl")
joblib.dump(numeric_cols, "model/numeric_cols.pkl")
joblib.dump(best_model_name, "model/best_model_name.pkl")

# Save category options for building the Streamlit input widgets
category_options = {col: sorted(df[col].unique().tolist()) for col in categorical_cols}
joblib.dump(category_options, "model/category_options.pkl")

results_df.to_csv("model/model_comparison.csv", index=False)

print("\nSaved model + preprocessing artifacts to model/")
