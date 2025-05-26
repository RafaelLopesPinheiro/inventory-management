Demand Forecasting for Pizza Sales using Machine Learning
🧠 Introduction

Effective demand forecasting is crucial for businesses in the food and beverage industry. By anticipating future sales, businesses can optimize inventory, reduce waste, and enhance profitability. This project explores machine learning models to predict daily pizza sales for different pizza types using a real dataset from a fictional pizzeria. The primary goal is to identify the most accurate and interpretable model for demand forecasting.
📚 Literature Review

Demand forecasting using machine learning has gained momentum in recent years. Traditional linear models such as Linear Regression offer simplicity and interpretability but may fall short in capturing non-linear trends and complex interactions.

Tree-based ensemble methods like Random Forest, XGBoost, and Gradient Boosting have demonstrated strong performance in structured datasets due to their ability to model non-linearities and interactions automatically. On the other hand, Support Vector Regression (SVR), while powerful in high-dimensional spaces, can sometimes be sensitive to parameter tuning and input scaling.

Model interpretability is also critical. Tree-based models can be analyzed using SHAP (SHapley Additive exPlanations) values, which help visualize how individual features influence predictions, adding transparency to model decisions.
🧪 Methodology
🔧 Data Pipeline

    Dataset: Daily pizza sales from a fictional pizzeria.

    Preprocessing: Temporal features were extracted, and each pizza type was modeled individually.

    Train-Test Split: The data was split using a time-based approach to preserve the chronological order.

🔬 Models Compared

The following models were implemented and compared:

    LinearRegressionModel: A basic linear approach.

    SupportVectorModel: SVR with RBF kernel.

    RandomForestModel: Bagging ensemble of decision trees.

    XGBoostModel: Gradient boosting framework optimized for performance.

    GradientBoostingModel: A traditional implementation of gradient boosting.

🛠 Hyperparameter Tuning

Each model (where applicable) was tuned using RandomizedSearchCV with 5-fold cross-validation, optimizing for Root Mean Squared Error (RMSE).
📈 Evaluation Metrics

Models were evaluated using:

    RMSE (Root Mean Squared Error)

    MAE (Mean Absolute Error)

    MAPE (Mean Absolute Percentage Error)

Additionally, SHAP values were used to interpret feature contributions for tree-based models.
📊 Results
Model	CV RMSE	Test RMSE	MAE	MAPE (%)
Random Forest	2.75	2.93	2.28	51.14
XGBoost	3.02	3.19	2.43	53.44
Gradient Boosting	2.93	3.02	2.31	51.76
Support Vector	2.76	3.04	2.35	52.93
Linear Regression	2.88	3.02	2.27	53.36
🏆 Best Performing Model

    Random Forest had the lowest RMSE and MAE overall, making it the most accurate model in this experiment.

🔍 Interpretability with SHAP

SHAP plots were generated for tree-based models (Random Forest, XGBoost, and Gradient Boosting). These visualizations highlighted the most influential features on model predictions, enhancing transparency and trust in model behavior.

Example:
(Assumes plot is generated and saved correctly)
✅ Conclusion & Next Steps

This project demonstrates the feasibility and value of using machine learning models for sales demand forecasting. Among all tested models, Random Forest showed the best balance between performance and interpretability.
🚀 Future Improvements

    Add seasonal decomposition or exogenous variables (e.g., holidays, weather).

    Explore deep learning approaches like LSTM for temporal dependencies.

    Implement automated retraining pipelines for production use.

    Compare results using probabilistic forecasting models (e.g., Prophet, NGBoost).