# Model Hyperparameters
CATBOOST_PARAMS = {
    'iterations': 1500,
    'learning_rate': 0.03,
    'depth': 7,
    'eval_metric': 'AUC',
    'early_stopping_rounds': 100,
    'auto_class_weights': 'Balanced',
    'verbose': 200,
    'allow_writing_files': False
}

XGB_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'auc',
    'early_stopping_rounds': 50,
    'verbosity': 1 
}

CAT_COLS = [
    'gender', 'ethnicity', 'education_level', 'income_level', 
    'smoking_status', 'employment_status', 'family_history_diabetes', 
    'hypertension_history', 'cardiovascular_history'
]

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
SUBMISSION_PATH = 'output/cv_ensemble_results.csv'