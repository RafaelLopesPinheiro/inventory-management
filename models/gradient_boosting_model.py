from sklearn.ensemble import GradientBoostingRegressor
from models.base_model import BaseModel


class GradientBoostingModel(BaseModel):
    def __init__(self):
        self.model = GradientBoostingRegressor()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
