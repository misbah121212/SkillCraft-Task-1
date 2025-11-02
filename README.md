🏠 House Price Prediction using Linear Regression

SkillCraft Technology – Machine Learning Internship (Task 01)

📘 Project Overview

The House Price Prediction project aims to predict the price of a house based on its various attributes such as living area, overall quality, year built, and other parameters.

This project demonstrates the complete end-to-end Machine Learning pipeline, from:

Data collection and preprocessing,

Exploratory Data Analysis (EDA),

Feature engineering and model training,

Performance evaluation, and

Deployment using a Streamlit web application.

The final model utilizes Linear Regression, a simple yet powerful supervised learning algorithm that captures the linear relationship between the dependent variable (Sale Price) and independent features.

🎯 Objective

To build a predictive model that accurately estimates house prices based on given input features and visualize the results interactively using Streamlit.

Key Learning Outcomes:

Understanding the fundamentals of regression analysis.

Performing feature selection and scaling.

Visualizing data relationships through plots.

Creating a web-based interactive model deployment.

📂 Dataset Description

The dataset used in this project is taken from Kaggle’s “House Prices – Advanced Regression Techniques” competition.

Dataset Link: House Prices Dataset on Kaggle

📄 Files Used:
File Name	Description
train.csv	Contains the training data including features and target variable SalePrice.
data_description.txt	Provides detailed descriptions of each feature in the dataset.
📊 Important Columns:
Feature	Description
LotArea	Lot size in square feet
OverallQual	Overall quality of materials and finish
GrLivArea	Above-ground living area (sq. ft.)
GarageCars	Number of cars in garage
TotalBsmtSF	Total area of basement (sq. ft.)
YearBuilt	Year of construction
SalePrice	Target variable – price of the house
🧩 Workflow
Step 1: Data Preprocessing

Importing libraries (pandas, numpy, matplotlib, seaborn, sklearn).

Loading dataset using Pandas.

Handling missing values and dropping irrelevant columns.

Encoding categorical features.

Splitting data into training and testing sets.

Step 2: Exploratory Data Analysis (EDA)

EDA helps understand the relationship between various features and the sale price.

Visualizations used:

Heatmaps for correlation analysis.

Scatter plots (e.g., GrLivArea vs. SalePrice).

Boxplots to detect outliers.

Step 3: Model Building

The model is built using Linear Regression from scikit-learn.

Steps:

Select numerical and relevant features.

Split data into 80% training and 20% testing sets.

Train the Linear Regression model.

Predict and evaluate using metrics like R² and MSE.

Step 4: Evaluation Metrics
Metric	Description
R² Score	Measures how well future samples are likely to be predicted.
Mean Squared Error (MSE)	Measures average of squared differences between predicted and actual values.

Expected Results:

R² Score: ~0.85

Low MSE value, indicating strong prediction accuracy.

Step 5: Visualization

Visualizations include:

Actual vs Predicted price plot.

Residual error distribution.

Feature correlation heatmap.

Step 6: Streamlit App Development

An interactive Streamlit web app allows users to:

Input property features (e.g., area, rooms, quality, etc.)

Predict the House Price instantly.

View a Matplotlib/Seaborn graph showing model accuracy and comparison.

🧠 Technologies & Libraries Used
Category	Tools/Packages
Programming Language	Python 3.10+
Libraries	Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib
Web Framework	Streamlit
IDE/Editor	VS Code / Jupyter Notebook
Dataset Source	Kaggle
📁 Project Structure
HousePricePrediction/
│
├── house_price_model.py       # Model training, EDA, and evaluation
├── app.py                     # Streamlit app for prediction and visualization
├── train.csv                  # Dataset file
├── data_description.txt       # Dataset documentation
├── house_price_model.pkl      # Saved trained model file
└── README.md                  # Project overview and documentation

⚙️ How to Run the Project
🔹 Step 1: Install Dependencies

Install the required libraries:

pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib

🔹 Step 2: Train the Model

Run the model training file:

python house_price_model.py


This will:

Load the dataset

Train the Linear Regression model

Display plots for analysis

Save the trained model as house_price_model.pkl

🔹 Step 3: Run the Streamlit App

Run your app with:

streamlit run app.py


Once launched, you can:

Input house features (area, quality, etc.)

View predicted price in real-time

See data visualizations in your app

🖼️ Application Preview
🎛️ Input Section

Users can select or input:

Living Area (GrLivArea)

Quality (OverallQual)

Garage (GarageCars)

Basement Area (TotalBsmtSF)

Year Built (YearBuilt)

📊 Output Section

Displays predicted house price.

Shows graph of actual vs predicted.

Displays distribution of residual errors.

📈 Sample Graph Output
🔸 Actual vs Predicted House Prices

A scatter plot visualizing the predicted prices compared to actual prices.
Closer points to the diagonal line = better performance.

🔸 Correlation Heatmap

Displays correlation between features and target (SalePrice) to identify key predictors.

🧮 Example Input & Output

Input:

Feature	Value
GrLivArea	1800
OverallQual	7
GarageCars	2
TotalBsmtSF	1200
YearBuilt	2005

Predicted Output:

Predicted House Price: $245,678

💾 Model Saving and Loading

After training, the model is saved as a .pkl file using Joblib:

import joblib
joblib.dump(model, 'house_price_model.pkl')


The Streamlit app loads it automatically for real-time predictions.

🧑‍💻 Developer Information

Name: Misba Sikandar
Internship: SkillCraft Technology – Machine Learning Internship
Task: Task 01 – House Price Prediction
Language: Python
IDE: Visual Studio Code
Duration: 1 Month Internship

📚 Learning Outcomes

By completing this task, I learned:

End-to-end development of a machine learning model.

Implementing regression using Scikit-learn.

Performing data analysis and visualization.

Deploying ML models using Streamlit.

Structuring a project for professional submission.

🏁 Conclusion

This project successfully demonstrates how Linear Regression can be applied to predict real-world prices of houses.
By integrating machine learning with a Streamlit interface, it becomes a complete product — simple, interpretable, and interactive.

This project fulfills Task 01 requirements for SkillCraft Technology’s Machine Learning Internship.
