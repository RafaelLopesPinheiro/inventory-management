# models/lightgbm_model.py
from lightgbm import LGBMRegressor
from models.base_model import BaseModel
import pandas as pd
import re

class LightGBMModel(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42):
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )

    def _sanitize_column_names(self, X):
        # Replace unsupported characters with underscores
        X = X.rename(columns=lambda col: re.sub(r'[\"\'{}\[\]:,]', '_', col))
        return X

    def train(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = self._sanitize_column_names(X.copy())
            # Convert object types to category if needed
            for col in X.select_dtypes(include='object').columns:
                X[col] = X[col].astype('category')
        else:
            raise ValueError("Input X must be a pandas DataFrame.")

        if pd.isnull(X).any().any() or pd.isnull(y).any():
            raise ValueError("Missing values detected in training data. Please handle them before training.")

        self.model.fit(X, y)
        self.feature_names_ = X.columns.tolist()

    def predict(self, X):
        # Apply same renaming during prediction
        X = self._sanitize_column_names(X.copy())
        return self.model.predict(X)
