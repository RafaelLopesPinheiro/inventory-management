import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from models.base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, input_shape, epochs=50, batch_size=16):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=self.input_shape))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X, y):
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        self.model.fit(X_reshaped, y_scaled, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        y_pred_scaled = self.model.predict(X_reshaped, verbose=0)
        return self.scaler_y.inverse_transform(y_pred_scaled).flatten()
