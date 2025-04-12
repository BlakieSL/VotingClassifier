# Voting Classifier Ensemble Demonstration

## Description
This project showcases an ensemble learning approach using scikit-learn's `VotingClassifier` to combine predictions from three distinct classifiers. The implementation demonstrates how model aggregation can improve performance on the non-linear `make_moons` dataset compared to individual models.

## Features
- Generates synthetic moons dataset with 10,000 samples and 0.4 noise
- Combines three classifier types in a hard-voting ensemble:
  - Support Vector Machine (RBF kernel)
  - Logistic Regression
  - Random Forest (100 trees)
- Visualizes unified decision boundary of the ensemble
- Compares combined model performance against individual component accuracies

## Installation
```bash
pip install numpy matplotlib scikit-learn
