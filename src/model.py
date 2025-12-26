# src/model.py
import sys
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

GLOBAL_SEED = 42

class EnsembleModel:
    def __init__(self, cb_params, xgb_params, cat_cols):
        self.cb_params = cb_params
        self.xgb_params = xgb_params
        self.cat_cols = cat_cols

    def run_cv(self, X, y, X_test):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
        val_scores = []
        test_preds_cat = np.zeros(len(X_test))
        test_preds_xgb = np.zeros(len(X_test))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            fold_seed = GLOBAL_SEED + fold
            
            print(f"\n" + "="*50)
            print(f"🚩 EXECUTING FOLD {fold+1} / 5 (Seed: {fold_seed})")
            print("="*50)
            sys.stdout.flush() # Force terminal to show the header immediately

            X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
            y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]

            # --- CatBoost Training ---
            print(f"\n🐱 Training CatBoost...")
            cb = CatBoostClassifier(**self.cb_params, random_seed=fold_seed)
            
            # EXPLICITLY pass verbose here to override any fit() defaults
            cb.fit(
                X_t, y_t, 
                eval_set=(X_v, y_v), 
                cat_features=self.cat_cols,
                verbose=self.cb_params.get('verbose', 200) # Ensure it uses your config
            )
            
            # --- XGBoost Training ---
            print(f"\n🏎️ Training XGBoost...")
            xgb = XGBClassifier(**self.xgb_params, random_state=fold_seed)
            xgb.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=100)

            # --- Blending & Scoring ---
            p_cat = cb.predict_proba(X_v)[:, 1]
            p_xgb = xgb.predict_proba(X_v)[:, 1]
            blend = (0.7 * p_cat) + (0.3 * p_xgb)
            
            fold_score = roc_auc_score(y_v, blend)
            val_scores.append(fold_score)
            print(f"\n✅ Fold {fold+1} Result: AUC = {fold_score:.4f}")

            test_preds_cat += cb.predict_proba(X_test)[:, 1] / 5
            test_preds_xgb += xgb.predict_proba(X_test)[:, 1] / 5

        return (0.7 * test_preds_cat) + (0.3 * test_preds_xgb), np.mean(val_scores)