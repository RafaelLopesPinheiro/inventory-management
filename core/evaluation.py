from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }


def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.values, label='Actual', marker='o')
    plt.plot(y_pred, label='Predicted', marker='x')
    plt.title(f'{title} - Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
