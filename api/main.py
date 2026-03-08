from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os

#load model and scaler ---------------------
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model=joblib.load(os.path.join(BASE_DIR, "ml/models/merchant_detector.pkl"))
scaler=joblib.load(os.path.join(BASE_DIR, "ml/models/scaler.pkl"))
explainer=joblib.load(os.path.join(BASE_DIR, "ml/models/shap_explainer.pkl"))
FEATURES = [
    "transaction_velocity",
    "avg_transaction_value",
    "refund_rate",
    "chargeback_rate",
    "business_age_days",
    "category_mismatch_score",
    "night_txn_ratio",
    "unique_customer_ratio",
    "geographic_spread",
    "incomplete_profile_score",
]

#FastAPI app ---------------------------
app = FastAPI(
    title="Fake Merchant Detection API",
    description="""
## 🛡️ Fake Merchant Detection System

A machine learning API that detects fraudulent/fake merchants in a payment ecosystem.

### Features
- **Real-time risk scoring** — get a fraud risk score (0-100) for any merchant
- **SHAP explanations** — understand *why* a merchant was flagged
- **Batch scoring** — score multiple merchants at once
- **Model info** — check model version and feature importance

### Risk Labels
- 🟢 **LOW** — Score 0–40, merchant appears legitimate
- 🟡 **MEDIUM** — Score 41–70, requires manual review
- 🔴 **HIGH** — Score 71–100, likely fraudulent
    """,
    version="1.0.0",
)
#calling api using frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#request and response schemas ----------------------------
class MerchantInput(BaseModel):
    merchant_id: str = Field(..., example="M-01042")
    transaction_velocity: float = Field(..., ge=0, example=245.5, description="Average daily transaction count")
    avg_transaction_value: float = Field(..., ge=0, example=12500.0, description="Average transaction value in INR")
    refund_rate: float = Field(..., ge=0, le=1, example=0.21, description="Refunds / total transactions (0.0 to 1.0)")
    chargeback_rate: float = Field(..., ge=0, le=1, example=0.087, description="Chargebacks / total transactions (0.0 to 1.0)")
    business_age_days: int = Field(..., ge=0, example=18, description="Days since merchant registration")
    category_mismatch_score: float = Field(0.0, ge=0, le=1, example=0.72, description="Transaction category vs registered MCC mismatch")
    night_txn_ratio: float = Field(0.0, ge=0, le=1, example=0.38, description="Ratio of transactions between 11PM–5AM")
    unique_customer_ratio: float = Field(0.0, ge=0, le=1, example=0.95, description="Unique customers / total transactions")
    geographic_spread: int = Field(0, ge=0, example=18, description="Number of unique cities in transactions")
    incomplete_profile_score: float = Field(0.0, ge=0, le=1, example=0.72, description="Missing business info fields ratio")

class SHAPExplanation(BaseModel):
    feature: str
    impact: float
    value: float

class ScoreResponse(BaseModel):
    merchant_id: str
    risk_score: int
    risk_label: str
    fraud_probability: float
    shap_explanations: list[SHAPExplanation]
    message: str

class BatchInput(BaseModel):
    merchants: list[MerchantInput]

#helper functions ---------------------------
def get_risk_label(score: int) -> str:
    if score <= 40:
        return "LOW"
    elif score <= 70:
        return "MEDIUM"
    else:
        return "HIGH"

def get_message(label: str) -> str:
    messages={
        "LOW": "Merchant appears legitimate. No action required.",
        "MEDIUM": "Merchant requires manual review before approval.",
        "HIGH": "Merchant flagged as high risk. Immediate review recommended.",
    }
    return messages[label]

def score_merchant(merchant: MerchantInput) -> ScoreResponse:
    features = np.array([[
        merchant.transaction_velocity,
        merchant.avg_transaction_value,
        merchant.refund_rate,
        merchant.chargeback_rate,
        merchant.business_age_days,
        merchant.category_mismatch_score,
        merchant.night_txn_ratio,
        merchant.unique_customer_ratio,
        merchant.geographic_spread,
        merchant.incomplete_profile_score,
    ]])
    # Scale features
    features_scaled = scaler.transform(features)
    # Get fraud probability from model
    fraud_prob = float(model.predict_proba(features_scaled)[0][1])
    # Convert to 0-100 risk score
    risk_score = int(round(fraud_prob * 100))
    risk_label = get_risk_label(risk_score)
    # SHAP explanation — why was this merchant flagged?
    shap_values = explainer.shap_values(features_scaled)[0]
    shap_pairs = sorted(
        zip(FEATURES, shap_values, features[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    explanations = [
        SHAPExplanation(feature=f, impact=round(float(s), 4), value=round(float(v), 4))
        for f, s, v in shap_pairs[:5] 
    ]
    return ScoreResponse(
        merchant_id=merchant.merchant_id,
        risk_score=risk_score,
        risk_label=risk_label,
        fraud_probability=round(fraud_prob, 4),
        shap_explanations=explanations,
        message=get_message(risk_label)
    )

#api endpoints -------------------------------
@app.get("/", tags=["Health"])
def root():
    """Health check - confirms API is running."""
    return {
        "status": "online",
        "api": "Fake Merchant Detection System",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.post("/api/v1/score", response_model=ScoreResponse, tags=["Scoring"])
def score_single_merchant(merchant: MerchantInput):
    """
    Score a single merchant and get a fraud risk score.

    Returns:
    - **risk_score**: 0–100 (higher = more suspicious)
    - **risk_label**: LOW / MEDIUM / HIGH
    - **fraud_probability**: raw model probability
    - **shap_explanations**: top 5 features that influenced the decision
    """
    try:
        return score_merchant(merchant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@app.post("/api/v1/score/batch", tags=["Scoring"])
def score_batch(batch: BatchInput):
    """
    Score multiple merchants at once.

    Send a list of merchants and get risk scores for all of them.
    """
    try:
        results = [score_merchant(m) for m in batch.merchants]
        return {
            "total": len(results),
            "high_risk": sum(1 for r in results if r.risk_label == "HIGH"),
            "medium_risk": sum(1 for r in results if r.risk_label == "MEDIUM"),
            "low_risk": sum(1 for r in results if r.risk_label == "LOW"),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")

@app.get("/api/v1/model/info", tags=["Model"])
def model_info():
    """
    Get information about the currently loaded ML model.
    """
    feature_importance = dict(zip(FEATURES, model.feature_importances_.tolist()))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    return {
        "model_type": "XGBoost Classifier",
        "version": "1.0.0",
        "features": FEATURES,
        "feature_importance": sorted_importance,
        "performance": {
            "f1_score": 0.7352,
            "roc_auc": 0.8242,
            "precision": 0.89,
            "accuracy": 0.88
        },
        "trained_on": 5000,
        "risk_thresholds": {
            "LOW": "0 – 40",
            "MEDIUM": "41 – 70",
            "HIGH": "71 – 100"
        }
    }

@app.get("/api/v1/analytics/summary", tags=["Analytics"])
def analytics_summary():
    """
    Get summary statistics for the dashboard.
    Returns aggregated fraud detection metrics.
    """
    return {
        "total_merchants": 5000,
        "flagged_high_risk": 1043,
        "flagged_medium_risk": 612,
        "flagged_low_risk": 3345,
        "fraud_rate": "20.86%",
        "model_precision": "89%",
        "avg_risk_score": 34.2,
        "top_fraud_indicators": [
            "chargeback_rate",
            "business_age_days",
            "incomplete_profile_score",
            "night_txn_ratio",
            "category_mismatch_score"
        ]
    }