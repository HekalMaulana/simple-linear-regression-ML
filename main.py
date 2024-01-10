# Import The Library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression

# Import The Dataset
dataset = pd.read_csv('sample_data/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Make variable regressor to call LinearRegression class
regresor = LinearRegression()

# Training the simple linear regression model on the training set
regressor.fit(X_train, y_train)
