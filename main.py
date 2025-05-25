from core.data_loader import DataLoader
from core.feature_engineering import FeatureEngineer
from core.evaluation import Evaluator, plot_predictions
from models.linear_regression import LinearRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.support_vector_model import SupportVectorModel
from models.gradient_boosting_model import GradientBoostingModel
from models.tuner import ModelTuner
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import shap
import numpy as np
import os


def run_pipeline():
    # Load and prepare data
    loader = DataLoader("data/Data Model - Pizza Sales.xlsx")
    df = loader.load()

    fe = FeatureEngineer(df)
    data = fe.transform()

    pizza_type = data.columns[1]  # Choose one pizza type
    y = data[pizza_type]
    X = data.drop(columns=['order_date', pizza_type])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    models = {
        "Random Forest": RandomForestModel(),
        "XGBoost": XGBoostModel(),
        "Gradient Boosting": GradientBoostingModel(),
        "Support Vector": SupportVectorModel()
    }

    best_models = {}
    scores = {}

    os.makedirs("plots", exist_ok=True)

    for name, model in models.items():
        print(f"\n--- Tuning {name} ---")
        best_model = ModelTuner.tune(model, X_train, y_train)
        best_models[name] = best_model

        # Cross-validation score
        cv_scores = cross_val_score(best_model.model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5)
        mean_rmse = -np.mean(cv_scores)
        scores[name] = mean_rmse
        print(f"Cross-validated RMSE for {name}: {mean_rmse:.2f}")


        # Final evaluation
        best_model.train(X_train, y_train)
        y_pred = best_model.predict(X_test)
        metrics = Evaluator.evaluate(y_test, y_pred)

        print(f"\n{name} Evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

        plot_predictions(y_test, y_pred, name)
        
        
        
        
        ## PLOT FOR EXPLAINABILITY IN TEST ## 
    
        if name in ["Random Forest", "XGBoost", "LightGBM"]:
            print(f"Generating SHAP values for {name}...")
            model = best_model.model  # Unwrap underlying model

            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)

            # Convert to shap.Explanation if needed
            shap.summary_plot(shap_values, X_train, show=False, plot_type="dot")
            plt.title(f"SHAP Summary Plot: {name}")
            # plt.savefig(f"plots/shap_{name.lower().replace(' ', '_')}.png")
            plt.close()

    # Plot comparison of models
    plt.figure(figsize=(10, 6))
    plt.bar(scores.keys(), scores.values(), color='skyblue')
    plt.ylabel("Mean RMSE")
    plt.title("Model Comparison (Cross-Validated RMSE)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/model_comparison_rmse.png")
    plt.show()


if __name__ == "__main__":
    run_pipeline()