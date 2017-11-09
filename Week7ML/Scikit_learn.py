import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from  sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('./daily_weather.csv')
#check all the columns
print(data.columns)
#print(data)
#if contains NaN values
print(data[data.isnull().any(axis = 1)])
#Data Cleaning
#del num cols
del data['number']
#drop null values
before_row = data.shape[0]
print(before_row) #how many of rows
data = data.dropna()
after_rows = data.shape[0]
print(after_rows) #after droped
print(before_row - after_rows)