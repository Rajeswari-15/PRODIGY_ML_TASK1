import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("train.csv")

# Feature engineering
data['HouseAge'] = data['YrSold'] - data['YearBuilt']
data['RemodelAge'] = data['YrSold'] - data['YearRemodAdd']
data['TotalBathrooms'] = data['FullBath'] + 0.5 * data['HalfBath']
data['HasGarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

# Define features and target
features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
    'TotalBathrooms', 'YearBuilt', 'HouseAge', 'RemodelAge',
    'TotRmsAbvGrd', 'HasGarage', 'Neighborhood'
]
target = 'SalePrice'

# Prepare data
data = data[features + [target]].dropna()
data = pd.get_dummies(data, columns=['Neighborhood'], drop_first=True)

X = data.drop(target, axis=1)
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt']
}

grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Model evaluation
print("\n📊 Model Evaluation:")
print("Best Params:", grid.best_params_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Cross-validation scores
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print("\n🔁 Cross-Validation Scores:", cv_scores)
print("Mean R²:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

# Feature importances
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plotting feature importances using Matplotlib
plt.figure(figsize=(10, 6))
plt.barh(importances['Feature'].head(10), importances['Importance'].head(10), color='skyblue')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
