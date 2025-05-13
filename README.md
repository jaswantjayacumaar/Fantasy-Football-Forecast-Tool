# Fantasy Football Forecast: Captain & Points Predictor

This project aims to predict player performance and assist users in making informed decisions for Fantasy Premier League (FPL). By using machine learning models trained on historical match data, the system can recommend the best captain picks and forecast player points.

## Features

- Predict whether a player will be in the starting 11
- Classify players based on expected points (high/low)
- Suggest optimal captain picks to maximize points
- Analyze player trends using features like form, xG, xA, FDR, and NT
- Explore multiple ML models (Naive Bayes, KNN, Random Forest, etc.)

## Models Used

- **Classification**: Naive Bayes, K-Nearest Neighbors, Random Forest
- **Regression**: Logistic Regression, Support Vector Machine
- **Future Scope**: Multi-layer Neural Networks, RNNs, LSTM

## Dataset

Player performance data sourced from public FPL APIs and datasets. Custom features were engineered to improve prediction accuracy.

## How to Run
### Requirements
Install necessary packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm
