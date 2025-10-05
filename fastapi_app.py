from fastapi import FastAPI
import joblib, json, pandas as pd

app = FastAPI(title="Stroke Prediction API", version="1.0.0")

MODEL_PATH = "models/stroke_model.joblib"
META_PATH  = "models/artifact.json"

model = joblib.load(MODEL_PATH)
meta  = json.load(open(META_PATH))
threshold = float(meta.get("threshold", 0.5))
features  = meta.get("features")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict(payload: dict):
    import pandas as pd
    x = pd.DataFrame([payload])[features]
    proba = model.predict_proba(x)[:,1][0]
    pred  = int(proba >= threshold)
    return {"probability": float(proba), "prediction": pred, "threshold": threshold}
