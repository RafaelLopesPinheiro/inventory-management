from sklearn.linear_model import LinearRegression
from models.base_model import BaseModel

class LinearRegressionModel(BaseModel):
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
