import pandas as pd
import numpy as np
import os
import time
from src.config import *
from src.features import FeatureEngineer
from src.model import EnsembleModel

def run():
    start_time = time.time()
    
    print("🚀 INITIALIZING DIABETES RISK PREDICTION PIPELINE")
    print("-" * 50)

    # 1. Loading Data
    if not os.path.exists(TRAIN_PATH):
        print(f"❌ Error: {TRAIN_PATH} not found.")
        return

    print(f"📦 Loading datasets from {os.path.dirname(TRAIN_PATH)}...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # 2. Engineering Features
    print("🧬 Engineering medical features and encoding variables...")
    engineer = FeatureEngineer(CAT_COLS)
    X = engineer.fit_transform(train.drop(columns=['id', 'diagnosed_diabetes']))
    y = train['diagnosed_diabetes']
    X_test = engineer.transform(test.drop(columns=['id']))
    print(f"✅ Feature matrix ready: {X.shape[0]} samples, {X.shape[1]} features.")

    # 3. Model Training & CV
    model = EnsembleModel(CATBOOST_PARAMS, XGB_PARAMS, CAT_COLS)
    final_preds, mean_auc = model.run_cv(X, y, X_test)

    # 4. Final Output Processing
    print("\n" + "!" * 50)
    print(f"🔥 FINAL CROSS-VALIDATION MEAN AUC: {mean_auc:.4f}")
    print("!" * 50)

    os.makedirs('output', exist_ok=True)
    
    # Process predictions to match your desired format
    submission = pd.DataFrame({
        'id': test['id'], 
        'Diabetes_Risk_Percentage': np.round(final_preds, 4)
    })
    
    submission.to_csv(SUBMISSION_PATH, index=False)
    
    duration = (time.time() - start_time) / 60
    print(f"\n📁 Results saved to: {SUBMISSION_PATH}")
    print(f"⏱️ Total Execution Time: {duration:.2f} minutes")
    print("-" * 50)

if __name__ == "__main__":
    run()