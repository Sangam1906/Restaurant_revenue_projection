import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_excel(r"D:\New folder\Tp_ML_model_rest.xlsx")

# Features and target
X = df.drop(columns=["Revenue"])
y = df["Revenue"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()


# 1️ Base Model (all features)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Full Model Performance")
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))


# 2 Positive-only Model

# Keep only numeric features with all positive values
positive_numeric = [col for col in numeric_features if (X[col] > 0).all()]
# Keep categorical as-is (they are non-numeric, no filtering needed)
positive_categorical = categorical_features.copy()

print("\nPositive numeric features:", positive_numeric)
print("Positive categorical features:", positive_categorical)

# Build model only if we have features
if len(positive_numeric) + len(positive_categorical) > 0:
    preprocessor_pos = ColumnTransformer([
        ("num", StandardScaler(), positive_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), positive_categorical)
    ])

    model_pos = Pipeline([
        ("preprocessor", preprocessor_pos),
        ("regressor", LinearRegression())
    ])

    model_pos.fit(X_train, y_train)
    y_pred_pos = model_pos.predict(X_test)

    print("\nPositive-only Model Performance")
    print("MSE:", mean_squared_error(y_test, y_pred_pos))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_pos)))
    print("R² Score:", r2_score(y_test, y_pred_pos))
else:
    print("\nNo positive-only features available. Skipping Positive Model.")

new_data = pd.DataFrame([{
    "Rest_Name": "Restaurant 8368",
    "Location": "Downtown",
    "Cuisine": "Indian",
    "Rating": 4,
    "Seating Capacity": 60,
    "Average Meal Price": 47,
    "Marketing Budget": 3218,
    "Social Media Followers": 36190,
    "Chef Experience Years": 10,
    "Number of Reviews": 523,
    "Avg Review Length": 174,
    "Ambience Score": 5,
    "Service Quality Score": 5,
    "Parking Availability": "No"
}])

# Predict revenue
predicted_revenue = model_pos.predict(new_data)
print("Predicted Revenue:", predicted_revenue[0])