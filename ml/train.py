#Training an XGBoost model to detect fake/fraudulent merchants
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import(
    classification_report, confusion_matrix, 
    roc_auc_score, f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

#setup -----------------------------
print("="*55)
print("FAKE MERCHANT DETECTION - MODEL TRAINING")
print("="*55)
# Create models folder if it doesn't exist
os.makedirs("ml/models", exist_ok=True)
# Features the model will use (everything except merchant_id and label)
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

#load data -------------------
print("\n[1/6] Loading dataset...")
df=pd.read_csv("data/merchants_dataset.csv")
print(f"Loaded {len(df)} merchants")
print(f"Legitimate : {(df['label']==0).sum()}")
print(f"Fraudulent : {(df['label']==1).sum()}")
X=df[FEATURES]
y=df["label"]

#train/test split -----------------------
print("\n[2/6] Splitting into train/test sets (80/20)...")
#ensures both splits have same fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size:{len(X_train)}")
print(f"Test size:{len(X_test)}")

#scale features --------------------------
print("\n[3/6] Scaling features...")
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
#save scaler
joblib.dump(scaler, "ml/models/scaler.pkl")
print("Scaler saved to ml/models/scaler.pkl")

#handle class imbalance with SMOTE ------------------
print("\n[4/6] Skipping SMOTE — using raw class distribution...")
X_train_resampled = X_train_scaled
y_train_resampled = y_train
print(f"Legit: {(y_train==0).sum()}, Fraud: {(y_train==1).sum()}")

#train XGBoost model -------------------
print("\n[5/6] Training XGBoost model...")
model = XGBClassifier(
    n_estimators=50,         # fewer trees
    max_depth=3,             # shallower trees — less powerful
    learning_rate=0.1,
    subsample=0.6,
    colsample_bytree=0.6,
    min_child_weight=10,     # needs more samples to make a split
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)
#save model
joblib.dump(model, "ml/models/merchant_detector.pkl")
print("Model saved to ml/models/merchant_detector.pkl")

#evaluate model ------------------------
print("\n[6/6] Evaluating model on test data...")
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # fraud probability
f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print(f"\n{'─'*45}")
print(f"  F1-Score  : {f1:.4f}  (target > 0.89)")
print(f"  ROC-AUC   : {auc:.4f}  (target > 0.95)")
print(f"{'─'*45}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"]))

#confusion matrix plot -------------------------
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,annot=True,fmt="d",cmap="Blues",
    xticklabels=["Legitimate", "Fraudulent"],
    yticklabels=["Legitimate", "Fraudulent"]
)
plt.title("Confusion Matrix - Fake Merchant Detection")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("ml/models/confusion_matrix.png",dpi=150)
print("Confusion matrix saved to ml/models/confusion_matrix.png")

#feature importance plot ---------------------
feat_importance=pd.Series(model.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(8,5))
feat_importance.plot(kind="barh", color="steelblue")
plt.title("Feature Importance - XGBoost")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("ml/models/feature_importance.png",dpi=150)
print("Feature importance saved to ml/models/feature_importance.png")

#SHAP explainer --------------------------
print("\n  Computing SHAP explainer (this may take ~30 seconds)...")
explainer = shap.TreeExplainer(model)
joblib.dump(explainer, "ml/models/shap_explainer.pkl")
print("  SHAP explainer saved to ml/models/shap_explainer.pkl")
# SHAP summary plot — shows which features matter most overall
shap_values = explainer.shap_values(X_test_scaled)
plt.figure()
shap.summary_plot(shap_values, X_test_scaled, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig("ml/models/shap_summary.png", dpi=150, bbox_inches="tight")
print("  SHAP summary plot saved to ml/models/shap_summary.png")

#done ---------------------------------
print(f"\n{'='*55}")
print("  ✅ TRAINING COMPLETE!")
print(f"  F1: {f1:.4f} | AUC: {auc:.4f}")
print("  All files saved in ml/models/")
print(f"{'='*55}\n")