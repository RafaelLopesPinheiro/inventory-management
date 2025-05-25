from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel
import numpy as np

class SupportVectorModel(BaseModel):
    def __init__(self, kernel='rbf', C=100, gamma=0.1, epsilon=0.1):
        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.scaler = StandardScaler()
        self._is_trained = False

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_trained = True

    def predict(self, X):
        if not self._is_trained:
            raise RuntimeError("You must call train() before predict()")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
