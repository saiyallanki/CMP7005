import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------
# 1. Load cleaned dataset
# -------------------------------------------------
df = pd.read_csv("combined_df.csv", parse_dates=["Date"])

# -------------------------------------------------
# 2. Feature engineering
# -------------------------------------------------
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["dayofweek"] = df["Date"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

df["City_le"] = df["City"].astype("category").cat.codes

# -------------------------------------------------
# 3. Select features and target
# -------------------------------------------------
features = [
    'PM2.5','PM10','NO','NO2','NOx','NH3',
    'CO','SO2','O3','Benzene','Toluene','Xylene',
    'City_le','year','month','day','dayofweek','is_weekend'
]

target = "AQI"

model_df = df.dropna(subset=[target])
X = model_df[features]
y = model_df[target]

# -------------------------------------------------
# 4. Train-test split (time-aware)
# -------------------------------------------------
train_mask = model_df["Date"] < "2019-01-01"
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

# -------------------------------------------------
# 5. Preprocessing pipeline
# -------------------------------------------------
numeric_features = features

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features)
])

# -------------------------------------------------
# 6. Random Forest pipeline
# -------------------------------------------------
model = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])

# -------------------------------------------------
# 7. Train model
# -------------------------------------------------
model.fit(X_train, y_train)

# -------------------------------------------------
# 8. Evaluate model
# -------------------------------------------------
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\nModel Evaluation (Local Training)")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# -------------------------------------------------
# 9. Save model (THIS FIXES YOUR ERROR)
# -------------------------------------------------
joblib.dump(model, "aqi_random_forest_model.pkl")
print("\nModel saved as aqi_random_forest_model.pkl")
