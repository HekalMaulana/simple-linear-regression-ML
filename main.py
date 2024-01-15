# Import The Library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import The Dataset
dataset = pd.read_csv('sample_data/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Make variable regressor to call LinearRegression class
regressor = LinearRegression()

# Training the simple linear regression model on the training set
regressor.fit(X_train, y_train)

# Predicting the test set result
y_pred = regressor.predict(X_test)

# Visualising the training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary Vs Experience (Training Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

# Visualising the test set result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary Vs Experience (Test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
