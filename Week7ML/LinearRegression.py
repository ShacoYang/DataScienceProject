'''
Linear Regression: captures relationship bwt numerical output and input variables
Relationship is modeled as linear
'''

import sqlite3
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

cnx = sqlite3.connect('C:\\Users\\yanlu\\Documents\\Data Science\\Week-7-MachineLearning\\Week-7-MachineLearning\\database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
df.head()
print(df.shape)
print(df.columns)
#Declar columns (Features)
features = [
       'potential', 'crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
       'gk_reflexes']
#Specify the prediction Target
target = ['overall_rating']

#Clean the data
df = df.dropna()
#Extract features and target values into Seperate DataFrames
x = df[features]
y = df[target]
#pick a typical row from features:
print(x.iloc[2])
print(y)

# split dataset into Training and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)
#perform two different modeling operations using different regression techniques
#1. select features and use a linear regressor to predict overall rating
regressor = LinearRegression()
#fine-tune parameters of the linear regressor
#to capture the interactions bwt two sets
#trying to fit x_train y_train and create a model
regressor.fit(X_train, y_train)
# OUTPUT: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
y_prediction = regressor.predict(X_test)
#what's the mean of the expected target value
y_test.describe()
#Evaluate Linear Regression Accuracy using Root Mean Square Error
#RMSE = 0 means no error
RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction))
print(RMSE)

#2. Decision Tree Regressor
regressor = DecisionTreeRegressor(max_depth=20)
regressor.fit(X_train, y_train)
y_prediction = regressor.predict(X_test)
y_prediction
y_test.describe()
RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction))
print(RMSE)