from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime
import joblib
import numpy as np
import os

from api.database import init_db, get_db, MerchantScore

# ── Load Model & Scaler ───────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model     = joblib.load(os.path.join(BASE_DIR, "ml/models/merchant_detector.pkl"))
scaler    = joblib.load(os.path.join(BASE_DIR, "ml/models/scaler.pkl"))
explainer = joblib.load(os.path.join(BASE_DIR, "ml/models/shap_explainer.pkl"))

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

# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fake Merchant Detection API",
    description="""
## 🛡️ Fake Merchant Detection System

A machine learning API that detects fraudulent/fake merchants in a payment ecosystem.

### Features
- **Real-time risk scoring** — get a fraud risk score (0-100) for any merchant
- **SHAP explanations** — understand *why* a merchant was flagged
- **Batch scoring** — score multiple merchants at once
- **Merchant history** — view all previously scored merchants
- **Model info** — check model version and feature importance

### Risk Labels
- 🟢 **LOW** — Score 0–40, merchant appears legitimate
- 🟡 **MEDIUM** — Score 41–70, requires manual review
- 🔴 **HIGH** — Score 71–100, likely fraudulent
    """,
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()
    print("✅ Database initialized")

# ── Schemas ───────────────────────────────────────────────────────────────────

class MerchantInput(BaseModel):
    merchant_id: str = Field(..., example="M-01042")
    transaction_velocity: float = Field(..., ge=0, example=245.5)
    avg_transaction_value: float = Field(..., ge=0, example=12500.0)
    refund_rate: float = Field(..., ge=0, le=1, example=0.21)
    chargeback_rate: float = Field(..., ge=0, le=1, example=0.087)
    business_age_days: int = Field(..., ge=0, example=18)
    category_mismatch_score: float = Field(0.0, ge=0, le=1, example=0.72)
    night_txn_ratio: float = Field(0.0, ge=0, le=1, example=0.38)
    unique_customer_ratio: float = Field(0.0, ge=0, le=1, example=0.95)
    geographic_spread: int = Field(0, ge=0, example=18)
    incomplete_profile_score: float = Field(0.0, ge=0, le=1, example=0.72)

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
    saved_to_db: bool = True

class BatchInput(BaseModel):
    merchants: list[MerchantInput]

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_risk_label(score: int) -> str:
    if score <= 30:   return "LOW"
    elif score <= 60: return "MEDIUM"
    else:             return "HIGH"

def get_message(label: str) -> str:
    return {
        "LOW":    "Merchant appears legitimate. No action required.",
        "MEDIUM": "Merchant requires manual review before approval.",
        "HIGH":   "Merchant flagged as high risk. Immediate review recommended.",
    }[label]

def score_merchant(merchant: MerchantInput, db: Session) -> ScoreResponse:
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

    features_scaled = scaler.transform(features)
    fraud_prob      = float(model.predict_proba(features_scaled)[0][1])
    risk_score      = int(round(fraud_prob * 100))
    risk_label      = get_risk_label(risk_score)

    shap_values = explainer.shap_values(features_scaled)[0]
    shap_pairs  = sorted(zip(FEATURES, shap_values, features[0]), key=lambda x: abs(x[1]), reverse=True)
    explanations = [
        SHAPExplanation(feature=f, impact=round(float(s), 4), value=round(float(v), 4))
        for f, s, v in shap_pairs[:5]
    ]

    # Save to database
    db_record = MerchantScore(
        merchant_id              = merchant.merchant_id,
        risk_score               = risk_score,
        risk_label               = risk_label,
        fraud_probability        = round(fraud_prob, 4),
        transaction_velocity     = merchant.transaction_velocity,
        avg_transaction_value    = merchant.avg_transaction_value,
        refund_rate              = merchant.refund_rate,
        chargeback_rate          = merchant.chargeback_rate,
        business_age_days        = merchant.business_age_days,
        category_mismatch_score  = merchant.category_mismatch_score,
        night_txn_ratio          = merchant.night_txn_ratio,
        unique_customer_ratio    = merchant.unique_customer_ratio,
        geographic_spread        = merchant.geographic_spread,
        incomplete_profile_score = merchant.incomplete_profile_score,
        score_at                = datetime.utcnow()
    )
    db.add(db_record)
    db.commit()

    return ScoreResponse(
        merchant_id       = merchant.merchant_id,
        risk_score        = risk_score,
        risk_label        = risk_label,
        fraud_probability = round(fraud_prob, 4),
        shap_explanations = explanations,
        message           = get_message(risk_label),
        saved_to_db       = True
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "api": "Fake Merchant Detection System",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.post("/api/v1/score", response_model=ScoreResponse, tags=["Scoring"])
def score_single(merchant: MerchantInput, db: Session = Depends(get_db)):
    """Score a single merchant and save result to database."""
    try:
        return score_merchant(merchant, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@app.post("/api/v1/score/batch", tags=["Scoring"])
def score_batch(batch: BatchInput, db: Session = Depends(get_db)):
    """Score multiple merchants and save all results to database."""
    try:
        results = [score_merchant(m, db) for m in batch.merchants]
        return {
            "total":       len(results),
            "high_risk":   sum(1 for r in results if r.risk_label == "HIGH"),
            "medium_risk": sum(1 for r in results if r.risk_label == "MEDIUM"),
            "low_risk":    sum(1 for r in results if r.risk_label == "LOW"),
            "results":     results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")

@app.get("/api/v1/merchants", tags=["Merchants"])
def get_all_merchants(limit: int = 50, risk_label: str = None, db: Session = Depends(get_db)):
    """Get all scored merchants from the database."""
    query = db.query(MerchantScore).order_by(MerchantScore.score_at.desc())
    if risk_label:
        query = query.filter(MerchantScore.risk_label == risk_label.upper())
    merchants = query.limit(limit).all()
    return {
        "total": len(merchants),
        "merchants": [
            {
                "id":                    m.id,
                "merchant_id":           m.merchant_id,
                "risk_score":            m.risk_score,
                "risk_label":            m.risk_label,
                "fraud_probability":     m.fraud_probability,
                "chargeback_rate":       m.chargeback_rate,
                "business_age_days":     m.business_age_days,
                "refund_rate":           m.refund_rate,
                "transaction_velocity":  m.transaction_velocity,
                "score_at":             m.score_at.strftime("%Y-%m-%d %H:%M:%S")
            }
            for m in merchants
        ]
    }

@app.get("/api/v1/merchants/{merchant_id}", tags=["Merchants"])
def get_merchant_history(merchant_id: str, db: Session = Depends(get_db)):
    """Get full scoring history for a specific merchant."""
    records = db.query(MerchantScore)\
                .filter(MerchantScore.merchant_id == merchant_id)\
                .order_by(MerchantScore.score_at.desc()).all()
    if not records:
        raise HTTPException(status_code=404, detail=f"Merchant {merchant_id} not found")
    return {
        "merchant_id":  merchant_id,
        "total_scores": len(records),
        "latest_score": records[0].risk_score,
        "latest_label": records[0].risk_label,
        "history": [
            {
                "risk_score":        r.risk_score,
                "risk_label":        r.risk_label,
                "fraud_probability": r.fraud_probability,
                "score_at":         r.score_at.strftime("%Y-%m-%d %H:%M:%S")
            }
            for r in records
        ]
    }

@app.get("/api/v1/model/info", tags=["Model"])
def model_info():
    """Get information about the currently loaded ML model."""
    feature_importance = dict(zip(FEATURES, model.feature_importances_.tolist()))
    sorted_importance  = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    return {
        "model_type":         "XGBoost Classifier",
        "version":            "2.0.0",
        "features":           FEATURES,
        "feature_importance": sorted_importance,
        "performance": {
            "f1_score":  0.7352,
            "roc_auc":   0.8242,
            "precision": 0.89,
            "accuracy":  0.88
        },
        "trained_on":    5000,
        "risk_thresholds": {
            "LOW":    "0 – 40",
            "MEDIUM": "41 – 70",
            "HIGH":   "71 – 100"
        }
    }

@app.get("/api/v1/analytics/summary", tags=["Analytics"])
def analytics_summary(db: Session = Depends(get_db)):
    """Get summary statistics from the live database."""
    total  = db.query(MerchantScore).count()
    high   = db.query(MerchantScore).filter(MerchantScore.risk_label == "HIGH").count()
    medium = db.query(MerchantScore).filter(MerchantScore.risk_label == "MEDIUM").count()
    low    = db.query(MerchantScore).filter(MerchantScore.risk_label == "LOW").count()
    return {
        "total_scored":        total,
        "flagged_high_risk":   high,
        "flagged_medium_risk": medium,
        "flagged_low_risk":    low,
        "fraud_rate":          f"{(high/total*100):.1f}%" if total > 0 else "0%",
        "model_precision":     "89%",
        "top_fraud_indicators": [
            "chargeback_rate",
            "business_age_days",
            "incomplete_profile_score",
            "night_txn_ratio",
            "category_mismatch_score"
        ]
    }

@app.delete("/api/v1/merchants/{merchant_id}", tags=["Merchants"])
def delete_merchant(merchant_id: str, db: Session = Depends(get_db)):
    """Delete all records for a specific merchant."""
    deleted = db.query(MerchantScore)\
                .filter(MerchantScore.merchant_id == merchant_id)\
                .delete()
    db.commit()
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Merchant {merchant_id} not found")
    return {"message": f"Deleted {deleted} record(s) for merchant {merchant_id}"}