import pandas as pd
movies = pd.read_csv('C:\\Users\\yanlu\\Documents\\Data Science\\Week-4-Pandas\\Week-4-Pandas\\movielens\\movies.csv', sep=',')
print(type(movies))
print(movies.head()) #top 15

# Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970

tags = pd.read_csv('C:\\Users\\yanlu\\Documents\\Data Science\\Week-4-Pandas\\Week-4-Pandas\\movielens\\tags.csv', sep=',')
tags.head()
#print(tags.head())

ratings = pd.read_csv('C:\\Users\\yanlu\\Documents\\Data Science\\Week-4-Pandas\\Week-4-Pandas\\movielens\\ratings.csv', sep=',')
#print(ratings.head())

# del tags['timestamp']
# del ratings['timestamp']

#extract 0th row: notice that it's affect a series
row_0 = tags.iloc[0]
print(type(row_0))
print(row_0)

print(row_0.index)
print(row_0['userId'])
print('rating' in row_0)
print(row_0.name)
row_0 = row_0.rename('first_row')
print(row_0.name)

# Extract row 0, 11, 2000 from DataFrame
print(tags.iloc[ [0,11,2000] ])