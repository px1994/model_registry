import os
import mlflow

from mlflow.tracking import MlflowClient

# -----------------------------
# DagsHub MLflow configuration
# -----------------------------

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/pritesh13590/model_registry.mlflow" #'http://localhost:5000'
os.environ['MLFLOW_TRACKING_USERNAME'] = "pritesh13590"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "d2eb6104e6b43c7ab7bfc01dafb923d335fb1054"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

client = MlflowClient()

model_name = "water_potability_rf"

model_version = 3

new_stage= "Production" # Staging 

client.transition_model_version_stage(
      name = model_name,
      version= model_version,
      stage= new_stage,
      archive_existing_versions=True  # for Staging False generally 

)

#print("Model moved to Staging")
print("Model moved to Production")