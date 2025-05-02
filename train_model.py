import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("insurance.csv")

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Column types
categorical_cols = ["sex", "smoker", "region"]
numerical_cols = ["age", "bmi", "children"]

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ("scale", StandardScaler(), numerical_cols)
])

# Create pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train pipeline
pipeline.fit(X, y)

# Save the pipeline
with open("data_pickle_2.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model pipeline saved as 'data_pickle_2.pkl'")
