#main.py

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


#linar_regression_simple.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("train.csv")

# Select only the requested features
features = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
target = data['SalePrice']

# Drop any rows with missing values in selected columns
data_clean = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].dropna()

X = data_clean[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data_clean['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Visualize Actual vs Predicted using Matplotlib
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Line for perfect prediction
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Linear Regression: Actual vs Predicted Prices")
plt.grid()
plt.show()

#Done with task1-House Price Prediction.

To walk through the video of my code and output.I have posted it on my linkedin  - https://www.linkedin.com/posts/yada-rajeshwari-022b8530b_prodigyinfotech-internship-task1completed-activity-7330210836130525184-czYE?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE8EndoBu65vprjIb-hNbUtHSPP2hiW1WU8
