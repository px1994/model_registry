import os
import mlflow

# -----------------------------
# DagsHub MLflow configuration
# -----------------------------

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/pritesh13590/model_registry.mlflow" #'http://localhost:5000'
os.environ['MLFLOW_TRACKING_USERNAME'] = "pritesh13590"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "d2eb6104e6b43c7ab7bfc01dafb923d335fb1054"


# model info
model_name = "water_potability_rf"
model_uri = "models:/m-bae2849a105642458917c686c4ff251e"


# register model
result = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print("Model registered successfully")
print(result)
