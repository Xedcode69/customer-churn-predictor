from fastapi import FastAPI
import joblib
from backend.schema import ChurnInput
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).resolve().parent / "random_forest_model.pkl"
model = joblib.load(MODEL_PATH)


@app.get("/")
def welcome():
    return "Welcome to churn predictor"


@app.post("/predict")
def predict(churn_input: ChurnInput):

    input_dict = churn_input.model_dump(by_alias=True)

    input_df = pd.DataFrame([input_dict])

    churn = model.predict(input_df)[0]

    return {"ChurnPrediction": int(churn)}
