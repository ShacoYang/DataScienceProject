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

#Descriptive Statistics
#select rating col
print(ratings['rating'].describe())
print(ratings.describe())
print(ratings['rating'].mean())
print(ratings.mean())
print(ratings['rating'].mode()) # find most frequent one

filter_1 = ratings['rating'] > 4
print(filter_1)
filter_1.any()

filter_2 = ratings['rating'] > 0
filter_2.all()

#Data Cleaning
movies.shape
#is any row null return boolean
movies.isnull().any()
ratings.shape
ratings.isnull().any()

tags.shape
tags.isnull().any()
tags = tags.dropna()

# %matplotlib inline
# ratings.hist(column = 'rating', figsize = (15,10))

#Transformation
#Slicing out cols
print(tags['tag'].head())
print(movies[['title','genres']].head())
ratings[1000:1010]
print(ratings[-10:]) #10 from the end
#value_counts let you find out the count of each unique value occurring in the input
tag_counts = tags['tag'].value_counts()
print(tag_counts)

#Filters for selection rows
is_highly_rated = ratings['rating'] >= 4.0
print(is_highly_rated[-5:])
#str.contains()
is_animation = movies['genres'].str.contains('Animation')
movies[is_animation][5:15]

#Aggregation across rows gives us big pics about the whole data set
ratings_count = ratings[['movieId','rating']].groupby('rating').count()
print(ratings_count)

average_rating = ratings[['movieId','rating']].groupby('movieId').mean()
average_rating.head()
#how many rating per movie
movie_count = ratings[['movieId','rating']].groupby('movieId').count()
movie_count.head()

movie_count = ratings[['movieId','rating']].groupby('movieId').count()
movie_count.tail()

# Merge Dataframes
print(tags.head())
print(movies.head())

t = movies.merge(tags, on = 'movieId', how='inner')
print(t.head());

#Combine aggreagation, merging, and filters to get useful analytics
#as_index = False generate new index
avg_ratings = ratings.groupby('movieId', as_index=False).mean()
del avg_ratings['userId']
print(avg_ratings.head())

box_office = movies.merge(avg_ratings, on='movieId', how='inner')
box_office.tail()

is_highly_rated = box_office['rating'] >= 4.0
box_office[is_highly_rated][-5:]

is_comedy = box_office['genres'].str.contains('Comedy')
box_office[is_comedy][:5]

box_office[is_comedy & is_highly_rated][-5:]

#split
#expend make sure it's an actual dataframe not just a series of lists
movie_genres = movies['genres'].str.split('|', expand=True)
movie_genres[:10]
#Add a new column for comedy genre flag
movie_genres['isComedy'] = movies['genres'].str.contains('Comedy')
print(movie_genres[:10])