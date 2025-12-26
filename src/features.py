import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols
        self.label_encoders = {}

    def fit_transform(self, df):
        df = self._apply_medical_logic(df)
        for col in self.cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        return df

    def transform(self, df):
        df = self._apply_medical_logic(df)
        for col in self.cat_cols:
            le = self.label_encoders[col]
            df[col] = le.transform(df[col].astype(str))
        return df

    def _apply_medical_logic(self, df):
        df_eng = df.copy()
        # AIP Index
        df_eng['AIP_Index'] = np.log10(df_eng['triglycerides'] / (df_eng['hdl_cholesterol'] + 1e-5))
        # Metabolic Risk
        df_eng['Metabolic_Risk_Score'] = (
            (df_eng['bmi'] > 30).astype(int) + 
            (df_eng['systolic_bp'] > 135).astype(int) + 
            (df_eng['triglycerides'] > 150).astype(int)
        ).astype(int)
        # Ratio
        df_eng['Sedentary_Ratio'] = df_eng['screen_time_hours_per_day'] / (df_eng['physical_activity_minutes_per_week'] / 60 + 1)
        return df_eng