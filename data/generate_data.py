import pandas as pd
import numpy as np
np.random.seed(42)
NUM_MERCHANTS=5000
FRAUD_RATIO=0.20
n_fraud=int(NUM_MERCHANTS*FRAUD_RATIO)
n_legit=NUM_MERCHANTS-n_fraud

def generate_legitimate(n):
    return {
        #avg ~50 transactions/day, normally distributed
        "transaction_velocity": np.random.normal(50,15,n).clip(1,200),

        #avg transaction value in INR-typically retail range
        "avg_transaction_value": np.random.normal(850,300,n).clip(50,5000),

        #refund rate: 1%-8% is normal for legit merchants
        "refund_rate": np.random.uniform(0.01,0.08,n),

        #chargeback rate: under 1.5% is healthy
        "chargeback_rate": np.random.uniform(0.001,0.015,n),

        #established businesses-typically 1-3 years old
        "business_age_days": np.random.normal(730,365,n).clip(30,3650),

        #category mismatch: low-merchant sells what they say they sell
        "category_mismatch_score": np.random.uniform(0.0,0.15,n),

        #night transactions: ~8% - some late orders, but mostly daytime
        "night_txn_ratio": np.random.normal(0.08,0.03,n).clip(0,0.25),

        #unique customer ratio: moderate - repeat customers exist
        "unique_customer_ratio": np.random.normal(0.55,0.15,n).clip(0.1,0.95),

        #Geographic spread: local to regional
        "geographic_spread": np.random.randint(1, 8, n),

        #Profile completeness: mostly complete
        "incomplete_profile_score": np.random.uniform(0.0, 0.15, n),

        "label": 0  #0 = legitimate
    }

def generate_fraudulent(n):
    return {
        "transaction_velocity":     np.random.normal(65, 30, n).clip(1, 200),
        "avg_transaction_value":    np.random.normal(800, 350, n).clip(50, 5000),
        "refund_rate":              np.random.uniform(0.03, 0.12, n),
        "chargeback_rate":          np.random.uniform(0.003, 0.025, n),
        "business_age_days":        np.random.normal(500, 400, n).clip(30, 3650),
        "category_mismatch_score":  np.random.uniform(0.05, 0.35, n),
        "night_txn_ratio":          np.random.normal(0.12, 0.06, n).clip(0, 0.35),
        "unique_customer_ratio":    np.random.normal(0.65, 0.15, n).clip(0.1, 0.95),
        "geographic_spread":        np.random.randint(1, 12, n),
        "incomplete_profile_score": np.random.uniform(0.05, 0.35, n),
        "label": 1
    }

def main():
    print("Generating merchant dataset...")

    #Generate both populations
    legit_df = pd.DataFrame(generate_legitimate(n_legit))
    fraud_df = pd.DataFrame(generate_fraudulent(n_fraud))

    #Combine and shuffle
    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # flip 12% of labels to create noise
    flip_idx = np.random.choice(len(df), size=int(len(df) * 0.12), replace=False)
    df.loc[flip_idx, "label"] = 1 - df.loc[flip_idx, "label"]

    #Add a merchant ID column
    df.insert(0, "merchant_id", [f"M-{str(i+1000).zfill(5)}" for i in range(len(df))])

    #Round floats for cleaner CSV
    float_cols=["transaction_velocity","avg_transaction_value","refund_rate","chargeback_rate","category_mismatch_score","night_txn_ratio","unique_customer_ratio","incomplete_profile_score"]
    df[float_cols]=df[float_cols].round(4)
    df["business_age_days"]=df["business_age_days"].astype(int)

    # Save to CSV
    output_path = "merchants_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"   Total merchants : {len(df)}")
    print(f"   Legitimate (0)  : {(df['label'] == 0).sum()} ({(df['label']==0).mean():.0%})")
    print(f"   Fraudulent (1)  : {(df['label'] == 1).sum()} ({(df['label']==1).mean():.0%})")
    print(f"\n   Columns: {list(df.columns)}")
    print(f"\n   Sample rows:")
    print(df.head(3).to_string())

if __name__ == "__main__":
    main()