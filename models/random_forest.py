from sklearn.ensemble import RandomForestRegressor
from models.base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
