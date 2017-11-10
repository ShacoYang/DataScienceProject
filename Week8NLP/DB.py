import os
data_iris_folder_content = os.listdir('./data/iris')
error_message = "Error: sqlite file not available, check instructions above to download it"
assert "database.sqlite" in data_iris_folder_content, error_message

import sqlite3
#Python module allow us to easy SQL operation
#connection Object
conn = sqlite3.connect('./data/iris/database.sqlite')
#Return a cursor for the connection
#cursor object
cursor = conn.cursor() # cursor obj is the interface to db
print(type(cursor)) #<class 'sqlite3.Cursor'>
#cursor.execute("SQL") to run SQL query
#type tuple
for row in cursor.execute("SELECT name FROM sqlite_master"):
    print(row)
#a shortcut to directly execute the query and gather the results is the fetchall method:
#type list
print(cursor.execute("SELECT name FROM sqlite_master").fetchall())
sample_data = cursor.execute("SELECT * FROM Iris Limit 3").fetchall()
print(sample_data)
# Because this is a list data structure
[row[0] for row in cursor.description]
#It is evident that the interface provided by sqlite3 is low-level,
# for data exploration purposes we would like to directly import data
# into a more user friendly library like pandas.
import pandas as pd

'''takes a SQL query and a connection object 
 imports the data into a DataFrame
keeping the same data types of the database columns
pandas provides a lot of the same functionality of SQL with a more user-friendly interface
'''
iris_data = pd.read_sql_query("SELECT * FROM Iris", conn)
print(type(iris_data))
print(iris_data.dtypes)

iris_setosa_data = pd.read_sql_query("SELECT * FROM Iris WHERE Species == 'Iris-setosa'", conn)

iris_setosa_data
print(iris_setosa_data.shape)
print(iris_data.shape)