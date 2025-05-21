# ⚽ Fantasy Football Forecast: Captain & Points Predictor

> A machine learning-based toolkit to help Fantasy Premier League (FPL) players make smarter choices by predicting player performance and suggesting optimal captain picks.

---

## 🚀 Features

* ✅ Predict if a player will start in the next match
* 📊 Classify players based on expected points (high/low)
* 👑 Suggest the best captain choices for the gameweek
* 🔍 Analyze form, Expected Goals (xG), Expected Assists (xA), Fixture Difficulty Rating (FDR), and Net Transfers (NT)
* 🧠 Explore various ML models (Naive Bayes, KNN, Random Forest, etc.)

---

## 🧠 Models Used

### 🔷 Classification

* Naive Bayes
* K-Nearest Neighbors (KNN)
* Random Forest

### 🔶 Regression

* Logistic Regression
* Support Vector Machine (SVM)

### 🔮 Future Scope

* Multi-layer Neural Networks (MLP)
* Recurrent Neural Networks (RNNs)
* Long Short-Term Memory (LSTM)

---

## 📁 Dataset

* Sourced from **official FPL APIs** and publicly available datasets
* Custom features were engineered for improved prediction accuracy

---

## 🛠️ How to Run

### 🔧 Requirements

Install all required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm
```

### 📂 File Structure

* `FPL.py` — Contains all models **except** Naive Bayes
* `FPL_Naive_Bayes.py` — Code specific to the **Naive Bayes** model
* `FPL_Prediction_Tool.ipynb` — **Jupyter Notebook** combining and executing all models with visualizations

---

## 📌 Future Improvements

* Incorporate player injury/suspension data
* Expanding and Diversifying Training Data
* Validation with Diverse Player Types
* Alternative Player Recommendations
