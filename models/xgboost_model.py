from xgboost import XGBRegressor
from models.base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42):
        self.model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    