# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
import pandas as pd
import joblib
import xgboost as xgb
from pydantic import BaseModel, Field, field_validator
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple
import asyncio
import os
import httpx

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom threshold to define No-Show
CUSTOM_THRESHOLD = 0.95

# Load the model and scaler (modify paths if necessary)
try:
    xgb_model: xgb.XGBClassifier = joblib.load("app/optimized_xgboost.pkl")
    scaler: StandardScaler = joblib.load("app/scaler.pkl")
    logger.info("Model and scaler loaded successfully.")
    xgb_model.set_params(predictor='cpu_predictor')
except Exception as e:
    logger.exception("Error loading the model or scaler")
    raise RuntimeError(f"Error loading the model or scaler: {e}")

app = FastAPI(
    title="No-Show Prediction API",
    description="API for predicting missed appointments",
    version="1.0"
)

# Limit concurrent requests
prediction_semaphore = asyncio.Semaphore(20)

class PatientData(BaseModel):
    gender: str
    age: int
    days_until_appointment: int
    distance_km: float
    wealth_level: str
    scholarship: bool
    hypertension: bool
    diabetes: bool
    alcoholism: bool
    handicap: bool
    sms_received: bool
    weekday: str

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        if value not in {"Male", "Female"}:
            raise ValueError("gender must be 'Male' or 'Female'")
        return value

    @field_validator("wealth_level")
    @classmethod
    def validate_wealth_level(cls, value: str) -> str:
        if value not in {"Low", "Medium", "High"}:
            raise ValueError("wealth_level must be 'Low', 'Medium', or 'High'")
        return value

    @field_validator("weekday")
    @classmethod
    def validate_weekday(cls, value: str) -> str:
        valid_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        if value not in valid_days:
            raise ValueError("weekday must be a valid day of the week")
        return value

# Mappings (used in preprocess_input if needed)
weekday_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}
wealth_mapping = {"Low": [1, 0], "Medium": [0, 1], "High": [0, 0]}
distance_mapping = [(0, 2, "Near"), (2, 10, "Medium"), (10, float("inf"), "Far")]
age_mapping = [(0, 18, "Minor"), (18, 35, "YoungAdult"), (35, 55, "Adult"), (55, float("inf"), "Senior")]

def preprocess_input(data: PatientData) -> pd.DataFrame:
    # Get the features used by the model
    try:
        model_features = xgb_model.get_booster().feature_names
    except Exception as e:
        logger.exception("Error retrieving model feature names")
        raise RuntimeError(f"Error retrieving model feature names: {e}")
    gender_m = 1 if data.gender == "Male" else 0
    weekday_encoded = [1 if weekday_mapping[data.weekday] == i else 0 for i in range(1, 7)]
    wealth_encoded = wealth_mapping[data.wealth_level]
    age_category = [1 if start <= data.age < end else 0 for start, end, _ in age_mapping]
    distance_category = [1 if start <= data.distance_km < end else 0 for start, end, _ in distance_mapping]
    features = {
        "Gender_M": gender_m,
        "Age": data.age,
        "Date.diff": data.days_until_appointment,
        "Distance_Hospital_km": data.distance_km,
        "Scholarship": int(data.scholarship),
        "Hipertension": int(data.hypertension),
        "Diabetes": int(data.diabetes),
        "Alcoholism": int(data.alcoholism),
        # Use "handicap" consistently
        "Handcap": int(data.handicap),
        "SMS_received": int(data.sms_received),
        "Last_Minute_Appointment": int(data.days_until_appointment <= 3)
    }
    features.update({f"Weekday_{i+1}": weekday_encoded[i] for i in range(len(weekday_encoded))})
    features.update({
        "Wealth_Level_Medium": wealth_encoded[0],
        "Wealth_Level_High": wealth_encoded[1]
    })
    features.update({f"Age_Group_{age_mapping[i][2]}": age_category[i] for i in range(len(age_mapping))})
    features.update({f"Distance_Category_{distance_mapping[i][2]}": distance_category[i] for i in range(len(distance_mapping))})
    df = pd.DataFrame([features])
    numerical_columns = ["Age", "Date.diff", "Distance_Hospital_km"]
    try:
        df[numerical_columns] = scaler.transform(df[numerical_columns])
    except Exception as e:
        logger.exception("Error scaling numerical data")
        raise RuntimeError(f"Error scaling numerical data: {e}")
    for feature in model_features:
        if feature not in df.columns:
            df[feature] = 0
    df = df[model_features]
    logger.info("Data preprocessing completed successfully.")
    return df

def model_predict(processed_data: pd.DataFrame) -> Tuple[int, float, str]:
    try:
        model_class = xgb_model.predict(processed_data)[0]
        probability = xgb_model.predict_proba(processed_data)[:, 1][0]
    except Exception as e:
        logger.exception("Error during model prediction")
        raise RuntimeError(f"Error during model prediction: {e}")
    final_prediction = "No-Show" if probability >= CUSTOM_THRESHOLD else "Show"
    return model_class, probability, final_prediction

@app.post("/predict")
async def predict(patient: PatientData, request: Request):
    try:
        async with prediction_semaphore:
            processed_data = await run_in_threadpool(preprocess_input, patient)
            model_class, probability, final_prediction = await run_in_threadpool(model_predict, processed_data)
    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
    return {
        "prediction": final_prediction,
        "probability_no_show": float(probability),
        "model_class_default_threshold": "No-Show" if int(model_class) == 1 else "Show"
    }

# Endpoint to return the updated dataset (ensuring that the file is the same one used by Streamlit)
@app.get("/updated_predictions")
async def updated_predictions():
    async def load_and_process_updated():
        # Ensure to use the same file "data/updated_dataset.csv"
        updated_file = "../medical_dashboard_streamlit/data/updated_dataset.csv"
        if not os.path.exists(updated_file):
            raise HTTPException(status_code=404, detail="Updated dataset not found.")
        df = await asyncio.to_thread(pd.read_csv, updated_file)
        df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], errors="coerce")
        df = df.dropna(subset=["AppointmentDay"])
        if "AppointmentTime" in df.columns:
            df["AppointmentTime"] = pd.to_datetime(df["AppointmentTime"], errors="coerce", format='%H:%M').dt.strftime("%H:%M")
            df["AppointmentTime"] = df["AppointmentTime"].fillna("-")
        reference_date = pd.Timestamp("today").normalize()
        df["days_from_ref"] = (df["AppointmentDay"] - reference_date).dt.days

        # For each row, call the prediction endpoint in parallel
        tasks = []
        for _, row in df.iterrows():
            patient_data = {
                "gender": "Male" if row.get("Gender", "M") == "M" else "Female",
                "age": int(row["Age"]),
                "days_until_appointment": int(row["days_from_ref"]),
                "distance_km": float(row["Distance_Hospital_km"]),
                "wealth_level": row["Wealth_Level"],
                "scholarship": bool(row["Scholarship"]),
                "hypertension": bool(row["Hipertension"]),
                "diabetes": bool(row["Diabetes"]),
                "alcoholism": bool(row["Alcoholism"]),
                "handicap": bool(row["Handcap"]),
                "sms_received": bool(row["SMS_received"]),
                "weekday": row.get("Weekday", "Monday")  # default if column is not present
            }
            tasks.append(call_prediction_api_async(patient_data))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        predictions = []
        for idx, result in enumerate(results):
            patient_id = df.iloc[idx]["PatientId"]
            if isinstance(result, Exception):
                predictions.append({
                    "PatientId": patient_id,
                    "Prediction": "Error",
                    "Probability_NoShow": None
                })
            else:
                predictions.append({
                    "PatientId": patient_id,
                    "Prediction": result.get("prediction", "N/A"),
                    "Probability_NoShow": result.get("probability_no_show", None)
                })
        return predictions

    try:
        predictions = await load_and_process_updated()
        return {"updated_predictions": predictions}
    except Exception as e:
        logger.exception("Error in /updated_predictions endpoint")
        raise HTTPException(status_code=500, detail=f"Error in /updated_predictions endpoint: {e}")

# Asynchronous function to call the /predict endpoint (useful for testing or internal use)
async def call_prediction_api_async(patient_data: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/predict", json=patient_data, timeout=10.0)
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
