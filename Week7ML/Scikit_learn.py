import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from  sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('./daily_weather.csv')
#check all the columns
print(data.columns)
'''
Index(['number', 'air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am',
       'avg_wind_speed_9am', 'max_wind_direction_9am', 'max_wind_speed_9am',
       'rain_accumulation_9am', 'rain_duration_9am', 'relative_humidity_9am',
       'relative_humidity_3pm'],
      dtype='object')
'''
#print(data)
#if contains NaN values
#print(data[data.isnull().any(axis = 1)])
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

clean_data = data.copy()
#(clean_data['relative_humidity_3pm'] > 24.99) -> True | False  * 1
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] > 24.99) * 1
print(clean_data['high_humidity_label'])
#store in Y
#if the values of clean data are more complex data types
#like lists, arrays then we need to use a deep copy
#.copy(deep = true)
y = clean_data[['high_humidity_label']].copy()
clean_data['relative_humidity_3pm'].head()
y.head()
##predict 9am --> humidity at 3pm
morning_features = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
        'rain_duration_9am']
X = clean_data[morning_features].copy()
X.columns
y.columns
##Fit on Train set
#how much data we want to test -> test_size
#train_test_split() takes TWO DataFrames and returns Four DataFrames
#x train, x test, y train, y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)

print(X_train)
print('====================')
# print(X_test)
print(X_train.describe())

'''
1. x is input data
2. y is label
3. create training sets for input and labels
4. create test set using input and labels
========================================
a model using training x and training y |
========================================
using test x to predict y (prediction)
in y_test the actual labels (actual)
'''
#max_leaf_nodes ->stop criteria for the tree induction
humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
#Create a decision tree classifier
#ckasifier tune itself to learn from the samples
humidity_classifier.fit(X_train, y_train) #-> decision tree-based classifier (a model)
print(type(humidity_classifier))
# ## Predict on Test set
# predictions = humidity_classifier.predict(X_test)
# print(predictions[:10])
# print(y_test['high_humidity_label'][:10])
# accuracy_score(y_true = y_test, y_pred = predictions)