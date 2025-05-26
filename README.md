Machine Learning Models for Pizza Demand Forecasting

1. Introduction

Demand forecasting plays a critical role in optimizing supply chains, inventory management, and staffing in the food and beverage industry. This project explores machine learning approaches to forecast pizza sales using a historical dataset. The objective is to build accurate models that can predict future demand for various pizza types, enabling better operational decisions.

2. Literature Review

Classical statistical models like ARIMA and exponential smoothing have long been used for time series forecasting. However, machine learning techniques have shown increasing effectiveness in capturing nonlinear patterns and complex dependencies. Random Forest, XGBoost, and Support Vector Regression (SVR) have proven useful for structured data prediction, while deep learning models like LSTM are particularly effective at modeling temporal sequences.

Model interpretability also plays a key role in model adoption. SHAP (SHapley Additive exPlanations) is used in this project to understand the influence of features in tree-based models.

3. Methodology

3.1 Data Preparation

    Source: Pizza sales dataset from an Excel spreadsheet.

    Preprocessing: Feature engineering includes date-based features and normalization.

    Split: The data is split into training and testing sets (80/20) without shuffling to preserve time-based order.

3.2 Models Tested

The following models were implemented in a modular, object-oriented structure:

    Linear Regression

    Random Forest

    Gradient Boosting

    XGBoost

    Support Vector Regression

    LSTM (Long Short-Term Memory Neural Network)

Each model is tuned (except LSTM) using RandomizedSearchCV to avoid overfitting while finding optimal hyperparameters.

3.3 Evaluation Metrics

The models are evaluated using:

    RMSE (Root Mean Squared Error)

    MAE (Mean Absolute Error)

    MAPE (Mean Absolute Percentage Error)

Cross-validation is used to validate model robustness. Visualizations of prediction and SHAP plots help explain the models' behavior.


4. Model Performance Results
Model	Cross-validated RMSE	RMSE	MAE	MAPE
Random Forest	2.75	2.93	2.28	51.14
XGBoost	3.02	3.19	2.43	53.44
Gradient Boosting	2.93	3.02	2.31	51.76
Support Vector	2.76	3.04	2.35	52.93
Linear Regression	2.88	3.02	2.27	53.36
LSTM	2.97	2.95	2.17	49.79


SHAP summary plots were generated for all tree-based models to interpret the impact of each feature on the model's predictions.
5. Conclusion and Next Steps

This project highlights how different machine learning models perform on demand forecasting tasks in the food industry. Random Forest and Gradient Boosting provided the best performance, balancing prediction accuracy and interpretability. The modular architecture allows easy testing of new models.
5.1 Next Steps:

    Fine-tune and evaluate the LSTM model further.

    Explore multi-output forecasting (predict multiple pizzas at once).

    Integrate weather, promotions, and holiday data for richer context.

    Deploy the best model in a real-time prediction API.