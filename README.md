# PRODIGY_ML_TASK1
Built a house price prediction model using Linear Regression and enhanced it with Random Forest, feature engineering, cross-validation, and hyperparameter tuning. Visualized results and feature importance effectively.

# 🏡 House Price Prediction - Internship Task 1

This project was completed as **Task 1 of my internship at Prodigy Infotech**. The goal was to build a machine learning model that predicts house prices based on various features such as square footage, number of bedrooms, bathrooms, and more.

## 📌 Problem Statement

> Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.

## 📂 Dataset

- Dataset Source: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Files Used:
  - `train.csv` – Training data
  - `test.csv` *(optional for testing/prediction)*
  - `data_description.txt` – Feature descriptions

## 🧠 Techniques Used

- **Linear Regression** – Initial implementation
- **Random Forest Regressor** – Final enhanced model
- **Feature Engineering**:
  - House age
  - Remodel age
  - Total bathrooms
  - Garage presence (binary)
- Hyperparameter Tuning - using `GridSearchCV`
- Cross-Validation - (5-fold) for robust evaluation
- Feature Importance - plot using `matplotlib`

## 📊 Evaluation Metrics:
- Mean Squared Error (MSE)
- R² Score
- Cross-Validation Scores (Mean & Std Dev)

## 📈 Visual Output
- Top 10 important features visualized using `matplotlib` bar chart.

## 🛠️ Tools & Libraries
- Python
- pandas
- numpy
- scikit-learn
- matplotlib

## 🔧 Installation

Install required libraries:
'''
pip install pandas numpy scikit-learn matplotlib
'''

## ▶️ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Rajeswari-15/house-price-prediction.git
   ```
2. Add the `train.csv` file to the root folder.
3. Run the script:

   ```bash
   python main.py
   ```

## 📽 Demo Video

Check out the demo output video on my [LinkedIn profile](https://linkedin.com/in/yada-rajeshwari-022b8530b).


## ✅ Project Status

✅ Task 1 successfully completed as part of my internship at **Prodigy Infotech**.
📌 Fully functional machine learning pipeline with evaluation and visualization.
