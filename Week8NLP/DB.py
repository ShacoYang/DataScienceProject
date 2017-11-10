import os
data_iris_folder_content = os.listdir('C:\\Users\\yanlu\\Documents\\Data Science\\Week-8-NLP-Databases\\Week-8-NLP-Databases\\data\\iris')
error_message = "Error: sqlite file not available, check instructions above to download it"
assert "database.sqlite" in data_iris_folder_content, error_message

import sqlite3
#Python module allow us to easy SQL operation
#connection Object
conn = sqlite3.connect('C:\\Users\\yanlu\\Documents\\Data Science\\Week-8-NLP-Databases\\Week-8-NLP-Databases\\data\\irisdatabase.sqlite')
#Return a cursor for the connection
#cursor object
cursor = conn.cursor()
print(type(cursor)) #<class 'sqlite3.Cursor'>