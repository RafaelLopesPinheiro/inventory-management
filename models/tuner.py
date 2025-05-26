from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
import numpy as np


class ModelTuner:
    @staticmethod
    def tune(model: BaseEstimator, X, y):
        class_name = model.__class__.__name__

        if class_name == "LSTMModel":
            return ModelTuner._tune_lstm(model, X, y)
        
        
        if hasattr(model, 'get_params'):
            if model.__class__.__name__ == "RandomForestModel":
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif model.__class__.__name__ == "XGBoostModel":
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            elif model.__class__.__name__ == "GradientBoostingModel":
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7]
                }
            elif model.__class__.__name__ == "SupportVectorModel":
                param_grid = {
                    'C': [0.1, 1, 10],
                    'epsilon': [0.1, 0.2, 0.3]
                }
            else:
                return model

            estimator = model.model
            search = RandomizedSearchCV(estimator, param_distributions=param_grid,
                                        n_iter=10, cv=3, scoring='neg_mean_squared_error',
                                        random_state=42, n_jobs=-1, verbose=0)
            search.fit(X, y)
            model.model = search.best_estimator_
            return model
        else:
            return model



    @staticmethod
    def _tune_lstm(model, X, y):
        best_model = None
        best_rmse = float("inf")
        best_config = None

        input_shape = (X.shape[1], 1)
        configs = [
            {'epochs': 20, 'batch_size': 16},
            {'epochs': 30, 'batch_size': 32},
            {'epochs': 50, 'batch_size': 64},
        ]

        for config in configs:
            candidate = model.__class__(input_shape=input_shape,
                                        epochs=config['epochs'],
                                        batch_size=config['batch_size'])
            candidate.train(X, y)
            y_pred = candidate.predict(X)
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = candidate
                best_config = config

        print(f"Best LSTM config: {best_config} with RMSE: {best_rmse:.2f}")
        return best_model