# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:22:20 2025

@author: JaswantJayacumaar
"""

#Naive Bayes and Accuracy comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset, select features and binarize target based on threshold
def load_and_prepare_data(file_path, threshold=8):
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[:, 15:26].values
    y = dataset.iloc[:, 0].values
    y = (y > threshold).astype(int)
    return train_test_split(X, y, test_size=0.25, random_state=0)

# Compute mean and std deviation for each feature per class
def compute_mean_std(X_train, y_train):
    means = {}
    stds = {}
    for c in np.unique(y_train):
        means[c] = X_train[y_train == c].mean(axis=0)
        stds[c] = X_train[y_train == c].std(axis=0)
    return means, stds

# Calculate Gaussian probability density (without normalization factor)
def gaussian_prob(x, mean, std):
    return np.exp(-((x - mean) ** 2) / (2 * std ** 2 + 1e-10))

# Predict class labels using Naive Bayes assumption
def predict_naive_bayes(X, means, stds, priors):
    y_pred = []
    for sample in X:
        posteriors = {}
        for c in priors:
            likelihood = np.prod(gaussian_prob(sample, means[c], stds[c]))
            posteriors[c] = priors[c] * likelihood
        y_pred.append(max(posteriors, key=posteriors.get))
    return np.array(y_pred)

# Load data and split into train/test sets
X_train, X_test, y_train, y_test = load_and_prepare_data('Dataset.csv')

# Train custom Naive Bayes model
means, stds = compute_mean_std(X_train, y_train)
priors = {c: np.mean(y_train == c) for c in np.unique(y_train)}
y_train_pred_custom = predict_naive_bayes(X_train, means, stds, priors)

# Train and predict with sklearn GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_test_pred_sklearn = gnb.predict(X_test)

# Print accuracy scores
print(f"Custom Naive Bayes Training Accuracy: {accuracy_score(y_train, y_train_pred_custom):.2f}")
print(f"Sklearn GaussianNB Test Accuracy: {accuracy_score(y_test, y_test_pred_sklearn):.2f}")

# Plot confusion matrices for train (custom NB) and test (sklearn NB)
cm_train = confusion_matrix(y_train, y_train_pred_custom)
cm_test = confusion_matrix(y_test, y_test_pred_sklearn)

disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=[0, 1])
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=[0, 1])

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
disp_train.plot(ax=axs[0], cmap='Blues')
axs[0].set_title("Custom NB - Train")
disp_test.plot(ax=axs[1], cmap='Greens')
axs[1].set_title("GaussianNB - Test")
plt.tight_layout()
plt.show()