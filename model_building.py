# model_building.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ---------------------------
# 1. Load the dataset
# ---------------------------
data = pd.read_csv("Dataset/Housing.csv")  # adjust path if needed

# ---------------------------
# 2. Feature Engineering
# ---------------------------
# ✅ Create price_per_area
data["price_per_area"] = data["price"] / data["area"]

# Drop redundant or weak features if any
# Example: if 'area' and 'price_per_area' both exist, sometimes one dominates
# but here we'll keep both for now

categorical_cols = ["mainroad", "guestroom", "basement", "hotwaterheating",
                    "airconditioning", "prefarea", "furnishingstatus"]

# OneHotEncode categorical columns
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first'), categorical_cols)],
    remainder='passthrough'
)

X = ct.fit_transform(data.drop("price", axis=1))
y = data["price"]

# Scale features (Random Forest doesn’t need scaling but for consistency we’ll keep it)
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 3. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------
# 4. Train Random Forest
# ---------------------------
model = RandomForestRegressor(
    n_estimators=500,   # more trees = better performance
    max_depth=20,       # deeper trees for capturing complexity
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------
# 5. Evaluation
# ---------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100

print("✅ Model Evaluation with Random Forest + Feature Engineering:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)
print("Accuracy: {:.2f}%".format(accuracy))

# ---------------------------
# 6. Save Model & Preprocessors
# ---------------------------
joblib.dump(model, "house_price_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(ct, "encoder.joblib")

print("🎉 Random Forest Model, Scaler & Encoder saved successfully with joblib!")
