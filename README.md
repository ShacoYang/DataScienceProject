# DataScienceProject
## Only useful if the insights can be turned into Actions and Action should be carefully defined and evaluated  
* ACQUIRE
    * Import raw dataset to analytics platform    
* PREPARE
    * Explore  Visulaize
    * Perform Data Cleaning
* ANALYZE 
    * Feature Selection
    * Model Selection
    * Analyze results
* REPORT
    * Predent findings
* ACT
    * Stakeholders use them

## Data Source
* Database
    - Relational
    - NoSQL
* Text Files
    - CVS
    - Text
* Live feeds
    - Sensors  
    - Online platforms  
        - Twitter, FB, ...
        
## Data Cleaning
* Missing entries
* Garbage values
* Nulls

## Analysis and Modeling
* Supervised learning
* Unsupervised learning
* Semi supervised learning

# Processing (Iterative process)
1. Acquire: Indetify data sets, Retrieve data, Query data  
2. Prepare:  
a. Explore (preliminary analysis, understand nature of data)   
b. Pre-process (Clean, Integrate, package)  
3. Analyze: build models
4. Report: Communicate Results
5. Act: Apply results

## Modeling: 
#### Selecting one of the following techniques:
##### 1.Classification:  
* Predict category(weather prediction)  
when model has to predict a numeric value instead of category --> regression  
##### 2.Regression:  
* Predict numeric value (stock price)
##### 3.Clustering: 
* Organize similar items into groups (Seniors, adults, Teen)  
##### 4.Association Analysis:  
* Find rules to capture associations between items
##### 5.Graph Analysis:
* Use graph structures to find connections between entities  


## Report:
#### Communicate result: what actions should follow  
1. Determining what part of the analysis is the most important to offer the biggest value  
(Punchline, main result)  

# Python basic
    * Numeric: integers, float, complex
    * Sequence: list, tuple, range
    * Binary: byte, bytearray
    * True/False: bool
    * Text: string
  
  
* Loop  
    * for i in range(start, stop) : 
    * start inclusive, stop exclusive(one few than stop)  
    ```python
    for i in range(2, 12, 3) :  
      print(i)
    2 5 8 11
   ```
   ```python
    i = 2
    while i < 12:
      print (i)
      i += 3
    ```   
* Conditions  
    ```python
    for i in range(0, 5):
      if i % 3 == 0:
          print (i)
      elif i % 3 == 1:
          print (i + 10)
      else:
          print (i - 10)
    ``` 
* Functions
    ```python
    def my_abs (val):
      if val < 0:
          return 0 - val
      return val
    print (my_abs(-5))
    x = my_abs(-10)
    print(x)
    5
    10
    None  
    ```
    
    ```python
    def inc_val(val):
      val  = val + 1
    x = 7
    inc_val(x)
    print (x)
    7 (x: 7 val : 8)
    ```
* Scope Rules  
Global Variable : Not recommend using Global variable
    ```python
    my_val = 0
    def my_abs(val):
        if val < 0:
            return 0 - val
        return val
    print (my_val)
    ``` 

#Data structures and Basic Libraries
### [String functions](https://docs.python.org/3/library/string.html)  
  * .lower() .upper()  
  * Concatenation: '1' + '2' --> '12'
  * Replication: '1' * 2 + '2' * 3 --> '11222'  
  * Strip: s.strip() --> get rid of the spaces and new line.  
  ```python
    strip(s,[,chars]) -> return a copy of the String with leading and trailing chars removed.
    s = '**10**'
    s.strip('*')
    '10'
    ```
   * Split
   * Slicing
   ```python
    H  E  L  L  O
    0  1  2  3  4
   -5 -4 -3 -2 -1
    word = "hello"
    word[1:3] --> 'ek'
    word[4:7] --> 'o'
    word[-4:-1] --> 'ell'
   ```
   * Substring test
   ```python
    word = 'Hello'
    'He' in word -> True
    word.find('el') -> 1
   ```
   * Convert to Number
   ```python
    word = '1234'
    int(word) -> 1234
    float(word) -> 1234.0
   ```  
### List in Python  (ArrayList in Java)
   ```python
    list = [11, 22, 33]
    list -> [1, 2, 3]
             0  1  2
    list[1] -> 22
    list[3] -> error
    
    for i in list: 
        print (i)
    for i in range (0, len(list)):
        print(list[i])
   ```
   * Append
   ```python
    list.append(44)
   ```
   * Delete (index)
   ```python
    list.pop(2)
   ```
   * Removing (value)
   ```python
    list.remove(33)
   ```
   * merge two list: extend  
   ```python
    list = [1,2]
    list2 = [4,5]
    list.extend(list2)
    list -> [1,2,4,5]
    list.append(list2)
    list -> [1,2,[4,5]]
   ```
   * Zipping List
   ```python
    list = [1,2]
    list2 = [4,5]
    for x, y in zip(list, list2):
        print (x, ", ", y)
    1, 4
    2, 5
   ```
### Tuples (immutable)  
```python
    tuple1 = ('a','b',5)
    tuple1 -> ('a', 'b', 5)
                0    1   2
    tuple1[1] -> 'b'
```
### Dictionary (Map in Java)
* Key has to be immutable
* Unordered
```python
    dict = {('Ghostbusters', 2016): 5,4, 
            ('Ghostbusters', 1984): 7.8}
    search: dict [('Ghostbusters', 2016)] -> 5.4
    len(dict) -> 2
    add: dict [('Cars', 2006)] = 7.1
    #safer way to get from dic: get / in
    dict.get(('Cars', 2006))
    ('Cars', 2006) in dict
```
* Deleting pop
* Iterating
```python
    #keys
    for i in dict:
        print (i)
    #bot keys and values
    for key, value in dict.items():
        print (key, ":", value)
```
*CAN'T mutate a dic object while iterating
* if if want to go through the entire data structure
and find everything in a criteria then remove those
```python
# first create an empty list
# iterate to the keys in the dic
# check if second element in the key tuple
# if so append the key to  to_remove list
# check -> append -> remove
    dict = {('Ghostbusters', 2016): 5,4, 
            ('Ghostbusters', 1984): 7.8}
    to_remove = []
    for i in dict:
        if (i[1] < 2000) :
            to_remove.append(i)
    for i in to_remove
        dict.pop(i)
```

### List comprehension
```python
 list = [ i ** 2 for i in range(1, 5)]
 list
 1, 4, 9, 16
 
 dict = { j: j ** 2 for j in range(1,5)}
 dict
 {1: 1, 2: 4, 3: 9, 4: 16}
```

### Set
* Unordered
* Unique
* Support set operations (union, intersection)
```python
loes_color = set (['blue', 'red', 'green'])
loes_color.add('yellow')
```
  * discard (do nothing if not in the set)
  ```python
    loes_colors.dicard('black')
    remove
  ```
  * Union (all unique items in two separate sets)
  ```python
    loes_color = set (['blue', 'red', 'green'])
    ilkays_color = set(['blue', 'yellow'])
    either = ilkays_color.union(loes_color) # set1 | set2
    either -> {'blue', 'red', 'green', 'yellow'}
  ```
  * Intersection
  ```python
    loes_color = set (['blue', 'red', 'green'])
    ilkays_color = set(['blue', 'yellow'])
    both = ilkays_color.intersection(loes_color) # set1 & set2
    both -> {'blue'}  
  ```
# Numpy: speed and Functionality
* Muti-dimensional Arrays
* Built-in array operations
* Simplified, powerful array interactions -> broadcasting
* Integration of other languages

* Reference or Copy
```python
an_array = np.array([[11,12,13,14],
                    [21,22,23,24],
                    [31,32,33,34]])
a_slice = an_array[:2 , 1:3]
#create a copy of the portion of np.array
a_slice = np.array(an_array[:2, 1:3])
```
* using the SINGLE INDEX is a SPECIAL CASE
```python
an_array = np.array([[11,12,13,14],[21,22,23,24],[31,32,33,34]])
row_rank1 = an_array[1, :]
#only a single []
print (row_rank1, row_rank1.shape)
'''(array([21, 22, 23, 24]), (4,))'''
row_rank2 = an_array[1:2, :]
print (row_rank2, row_rank2.shape)
'''(array([[21, 22, 23, 24]]), (1, 4))'''
```
* filter
```python
an_array = np.array([[11,12],[21,22],[31,32]])
#create a filter which will be boolean values
filter = (an_array > 15)
print(filter)
#select elements which meet the criteria
print(an_array[filter])
#for short,
an_array[(an_array > 20) & (an_array < 30)]
#change the element based on condition
an_array[an_array % 2 == 0] += 100
```
* Datatyeps and Operator 
```python
ex1 = np.array([11,12])
print(ex1.dtype) --> int64
print(type(ex1)) --> <class 'numpy.ndarray'>
```
* Arithmetic Operations:
```python
x = np.array([[111,112],[121,122]], dtype=np.int64)
y = np.array([[211.1,212.2],[221.1,222.1]], dtype=np.float64)
print(x)
print(y)
print(x + y)
#same as the numpy function "add"
print(np.add(x, y))
print(x - y)
print(np.subtract(x,y))
```
* Statistical, Sorting, Set operations
```python
# random 2 * 4 matrix
arr = 10 * np.random.randn(2,5)
print(arr)
# mean for all elements
print(arr.mean())
# mean by row
print(arr.mean(axis=1))
# mean by column
print(arr.mean(axis=0))
# sum
print(arr.sum())
#compute the medians
print(np.median(arr, axis=1))
```
* Sorting
    * Copy and Sort  
    ```python
    unsorted = np.random.randn(5)
    print(unsorted)
    # create copy and sort
    sorted = np.array(unsorted)
    sorted.sort()
    print(sorted)
    print(unsorted)
    ```
    * Inplace sorting  
    ```python
    #inplace sorting
    unsorted.sort()
    print(unsorted)
    ```
* Finding Unique elements:
```python
array  = np.array([1,2,3,4,1,2,4,2])
print(np.unique(array))
```
* Set Operations with np.array data type
```python
s1 = np.array(['desk','chair','bulb'])
s2 = np.array(['lamp', 'bulb','chair'])
print(s1, s2)
print(np.intersect1d(s1,s2))
print(np.union1d(s1,s2))
# element in s1 are not in s2
print(np.setdiff1d(s1,s2))
print(np.setdiff1d(s2,s1))

#whether each element in the array or not
#boolean
print(np.in1d(s1, s2))
```
* Broadcasting
```python
start = np.zeros((4,3))
print(start)
#create a rank 1 ndarry with 3 value
add_rows = np.array([1,0,2])
print(add_rows)
# add to each row of 'start' using broadcasting
y = start + add_rows
print(y)

#create an ndarray 4 * 1 to broadcast across colums
add_cols = np.array([[0,1,2,3]])
add_cols = add_cols.T #transpose on it, denoted by T
print(add_cols)
# add to each column of 'start'
y = start + add_cols
print(y)
```
```python
#broadcast in both dimensions
add_scalar = np.array([1])
print(start + add_scalar)
```
```python
# b1 b2
a = np.array([[0,0],[0,0]])
b1 = np.array([1,1])
b2 = 1
print(a + b1 == a + b2)
print(a + b2)
```
#Satellite Image Analysis using numpy
###Data source: Satellite Image from WIFIRE Project
* Loading libs: numpy, scipy, matplotlib  
```python
%matplotlib inline
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
```
* Creating a numpy array from an image file
```python
#choose a image file as an ndarray and display type
from skimage import data
photo_data = misc.imread('./wifire/sd-3layers.jpg')
type(photo_data)
```
```python
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
```
```python
photo_data.shape ->(3725, 4797, 3)
#it is a three layered matrix
#length width, the third number 3 is for three layers
#Red: Altitude
#Green: Aspect
#Blue: Slope
print(photo_data) 
```
* Pixel on the xth Row and xth Column
```python
photo_data[150,250] #RGB color for this pixel
photo_data[150,250,1] # Green
```
* Set a Pixel to ALL Zeros
```python
photo_data[150,250] = 0
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
```
* Changing colors in a Range
```python
photo_data = misc.imread('./wifire/sd-3layers.jpg')
#set GREEN LAYER for rows 200 to 800 to full intensity
photo_data[200:800, : ,1] = 255
#set 200-800 ALL LAYERS to white
#photo_data[200:800, :] = 255
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
```

#Pandas
[Panadas Documentation](http://pandas.pydata.org/pandas-docs/stable/)   
panda builds up Numpy, it enables **ingestion and manipulation**,  
it also enables **combining large data sets** using **merge** and **join**.  
Efficient library for **breaking datasets**, **transforming**, **recombining**. 
**Visualizations**  
* Pandas Series
    * 1d array-like object: many ways to index data  
    * Acts like an ndarray
* Pandas DataFrame
    * 2d elastic data structure  
    * Support heterogeneous data  
##### Series
```python
#Format -> pd.Series(data= [], index = [])
ser = pd.Series(data=[100,200,300], index=['tom','bob','nancy'])
#Data can be HETEROGENEOUS
ser = pd.Series([100, 'foo', 300, 'bar', 500], ['tom', 'bob', 'nancy', 'dan', 'eric'])
print(ser)
print(ser.index)
#Value of the location index by ''
print(ser.loc[['nancy','bob']])
#also work for index
print(ser[[2,3]])
#if an index exists
print('bob' in ser)
# string also *2 duplicate
print(ser * 2)
```
##### DataFrame
* 2-d labeled data structure
```python
#create a dictionary
d = {'one' : pd.Series([100., 200., 300.], index=['apple', 'ball', 'clock']),
     'two' : pd.Series([111., 222., 333., 4444.], index=['apple', 'ball', 'cerill', 'dancy'])}
df = pd.DataFrame(d)
print(df)
          one     two
apple   100.0   111.0
ball    200.0   222.0
cerill    NaN   333.0
clock   300.0     NaN
dancy     NaN  4444.0
```
* index, column
```python
df.index
#Index(['apple', 'ball', 'cerill', 'clock', 'dancy'], dtype='object')
df.colunms
#Index(['one', 'two'], dtype='object')
#give dic as the input and pick indices '...' for col indexes from the series
pd.DataFrame(d, index=['dancy', 'ball', 'apple'])
         one     two
dancy    NaN  4444.0
ball   200.0   222.0
apple  100.0   111.0
#select cols not exists
pd.DataFrame(d, index=['dancy', 'ball', 'apple'], columns=['two', 'five'])
          two five
dancy  4444.0  NaN
ball    222.0  NaN
apple   111.0  NaN

# data array of 2d,
# alex': 1, 'joe': 2 in 1
#'ema': 5, 'dora': 10, 'alice': 20 in 2
data = [{'alex': 1, 'joe': 2}, {'ema': 5, 'dora': 10, 'alice': 20}]
data
[{'alex': 1, 'joe': 2}, {'alice': 20, 'dora': 10, 'ema': 5}]
pd.DataFrame(data)
alex	alice	dora	ema	joe
0	1.0	NaN	NaN	NaN	2.0
1	NaN	20.0	10.0	5.0	NaN
#set index name
pd.DataFrame(data, index=['a','b'])
#selecct some of the elements from the dic as cols
pd.DataFrame(data, columns=['joe','dora'])
#set index name and select some elements
pd.DataFrame(pd.DataFrame(data,index=['a','b']), columns=['joe', 'dora','alice'])
```
* Basic DataFrame Operation
```python
df['three'] = df['one'] * df['two']
         one     two    three
apple   100.0   111.0  11100.0
ball    200.0   222.0  44400.0
cerill    NaN   333.0      NaN
clock   300.0     NaN      NaN
dancy     NaN  4444.0      NaN

df['flag'] = df['one'] > 250
          one     two    three   flag
apple   100.0   111.0  11100.0  False
ball    200.0   222.0  44400.0  False
cerill    NaN   333.0      NaN  False
clock   300.0     NaN      NaN   True
dancy     NaN  4444.0      NaN  False
three = df.pop("three") #remove col three
del df['two']
df.insert(2, 'copy_of_one', df['one'])
#get the first two values-->[:2] in data frame column 'one'
#assign it to 'one_upper_half' col
df['one_upper_half'] = df['one'][:2]
```
#### Data Ingestion:  
* CSV: input-> Pandas DataFrame obj containing contents of the file
* JSON: -> DataFrame or a Series containing the contents
* HTML: -> a list of Pandas DataFrames
* SQL: SQL Query, DB connection, DataFrame obj containing contents of the file  

#### Data Structures : [Series](https://github.com/ShacoYang/DataScienceProject/blob/master/Week4Pandas/Movies.py)
```python
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
tags.iloc[ [0,11,2000] ]
```
* describe()
    * data_frame.describe()
    * shows summary statistics of the dataframe
* corr()
    * data_frame.corr()
    * computes pairwise Pearson coefficient(p) of columns
    * other coefficients vailable: Kendall, Spearman
* func = min(), max(), mode(), median()
    * data_frame.func()
    * Frequently used optional parameter:
        * axis = 0(rows) or 1(columns)
* mean()
    * data_frame.mean(axis={0 or 1})
        * Axis = 0: index
        * Axis = 1: columns
    * Output Series or DataFrame with the mean values
* std()
    * data_frame.std(axis={0 or 1})
      * Axis = 0: index
      * Axis = 1: columns
    * Series or DataFrame with the Standard Deviation value
* any() all()
    * Returns whether ANY element or ALL elements are True
```python
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
```

####Data Cleaning
######Why cleaning making data ready for analysis  
   * Missing Values  
   * Outliers in the data  
   * Invalid data(<0 for age)  
   * NaN (np.nan)  
   * None value     
######Handle Data Quality Issues  
   * Replace the value
   * Fill gaps forward / backward
   * Drop fields
   * Interpolation
```python
#globally change values in a DataFrame
df.replace(9999.0,0) #replace all 9999 to 0
#forward backword fill gaps
df.fillna(method = 'ffill') # going down
df.fillna(method = 'backfill') # up
#interpolation
df.interpolate()
```
####Data Visualization  
* df.plot.bar()
    * Each col is represented by a diff col and turned into a bar  
* df.plot.box()
    * showing data distribution, each box has min and max and medium for colms
* df.plot.hist()
    * distribution of data and it can show skewness on unusual dispersion
* df.plot()
    * create quick line graphs of data sets
##### Frequent Data Operations  
* Slice out cols
    * df['col_name']
* Filter out rows
    * df[df['col_name']>0] --> select row where col_name is positive
* Insert New Column
    * df['col_name4'] = df['col_name3'] ** 2
* Add a new Row
    * df.loc[10] = [11,12]
* Delete a Row
    * df.drop(df.index[[5]])
* Delete a Col
    * del df['col_name']
* Group by and Aggregate: combine statistics about the DataFrame
    * df.groupby('student_id').mean() group by using studentID and extract mean scores for each subject
####Transformation
```python
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
```

#####[Merging DataFrames](http://pandas.pydata.org/pandas-docs/stable/merging.html)   
* pd.concat([left, left])
    * Stack Dataframes (Vertically)
    * pd.concat([left,right]) if some cells for cols didn't exist NaN or Missing values
* Inner Join using pandas.concat() (Horizontally)
    * pd.concat([left,right], axis = 1, join = 'inner')
* Stack Dataframes using append()  vertically
    * left.append(right)
* Inner join using merge() it can remove the duplicates
    * pd.merge(left, right, how = 'inner')
    
####Combine aggreagation, merging, and filters to get useful analytics
```python
#as_index = False generate new index
avg_ratings = ratings.groupby('movieId', as_index=False).mean()
del avg_ratings['userId']
print(avg_ratings.head())
#both filters  comedy movie and high rated
box_office[is_comedy & is_highly_rated][-5:]
```

#####Frequent String Operations
* str.split()
* str.contains() dtype: bool
* str.replace()
* str.extract() return **first** match found
```python
#split
#expend make sure it's an actual dataframe not just a series of lists
movie_genres = movies['genres'].str.split('|', expand=True)
movie_genres[:10]
```

###Summary 
* Data ingestion: how to ingest data in muti-formats, basic read opertions
* Series and DataFrame
* Func perform basic statistical operations on Series and DataFrame
    * describe()
    * min(), max()
    * std()
    * mode()
    * corr()
    * any(), all()
* Data Preparation
    * Detection
        * isnull()
        * any()
    * Cleaning
        * dropna()
* Data Visualization
    * inline plotting
    * Histograms
    * Boxplots
    * Changing limits on Y-axis
* Data Transformation
    * Slicing Colms
    * Filtering Rows
    * groupby()
        * mean()
        * count()
* Merging DataFrames  
    * merge()
        * how = inner
        * on = keys     
* String Operations
    * str.split()
    * str.contains()
    * str.extract()

##Data Visualization
#####Conceptual or data-driven
#####Declarative or exploratory  
* Good data visualization:
    * Trustworthy
        * Data presented is honestly portrayed
    * Accessible
    * Elegant
        * focus on relevant
####Matplotlib
* plotting Lib for Python
* Other libs:
    * Seaborn
    * ggplot
    * Altair
    * Bokeh
    * Plotly
    * Folium
```python
#how many unique countries
countries = data['CountryName'].unique().tolist()
print(len(countries))

#are there same number of country codes
# How many unique country codes are there ? (should be the same #)
countryCodes = data['CountryCode'].unique().tolist()
print(len(countryCodes))
# How many unique indicators are there ? (should be the same #)
indicators = data['IndicatorName'].unique().tolist()
len(indicators)
# How many years of data do we have ?
years = data['Year'].unique().tolist()
len(years)
#range of years
print(min(years), "to", max(years))
```
#####Matplotlib: Basic Plotting
* Chart Type
* Axes data ranges
* Axes labels
* Figure labels
* Legend
* Aesthetics
* Annotations
```python
##Plotting in matplotlib
# pick USA and indicator CO2 emissions
hist_indicator = 'CO2 emissions \(metric'
hist_country = 'USA'
mask1 = data['IndicatorName'].str.contains(hist_indicator)
mask2 = data['CountryCode'].str.contains(hist_country)
#stage is indicators matching the USA for country code and CO2 emissions
stage = data[mask1 & mask2]
print(stage.head())
```
* years and co2 emissions send to bar plot
    ```python
    years = stage['Year'],values
    CO2 = stage['Value'].values
    #create
    plt.bar(years, CO2)
    plt.show()
    ```
* Improve plot a bit
    ```python
    #Improve graph
    #a line plot
    plt.plot(stage['Year'].values, stage['Value'].values)
    #Label the axes
    plt.xlabel('Year')
    plt.ylabel(stage['IndicatorName'].iloc[0])
    #Label the figure
    plt.title('CO2 Emissions in the USA')
    plt.axis([1959,2011,0,25])
    plt.show()
    ```
* Histograms to explore the distribution of values
    ```python
    ##Using Histograms to explore the distribution of values
    hist_data = stage['Value'].values
    print(len(hist_data))
    #histogram of the data
    plt.hist(hist_data, 10, normed=False, facecolor='green')
    plt.xlabel(stage['IndicatorName'].iloc[0])
    plt.ylabel('# of Years')
    plt.title('Histogram Example')
    plt.grid(True) # add grid
    plt.show()
    ```
* USA relates to other contries
    ```python
    #plot a histogram of the emissions per capita
    # subplots returns a touple with the figure, axis attributes.
    fig, ax = plt.subplots()
    ax.annotate("USA",
                xy=(18, 5), xycoords='data',
                xytext=(18, 30), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"),
                )
    plt.hist(co2_2010['Value'], 10, normed=False, facecolor='green')
    
    plt.xlabel(stage['IndicatorName'].iloc[0])
    plt.ylabel('# of Countries')
    plt.title('Histogram of CO2 Emissions Per Capita')
    
    #plt.axis([10, 22, 0, 14])
    plt.grid(True)
    plt.show()
    ```

# Machine Learning  
used to build models to discover hidden patterns ad trends in the data,
allowing for data driven decisions to be made
* learning form data
* on its own
* used to discovering hidden patterns
* data-driven decisions

* Credit card fraud detection
* Handwritten digit recognition

#### Categories 
* classification: predict category
* regression: predict numeric value
* cluster analysis: organize similar items into groups
* association: find rules to capture associations between items

#### Supervised VS Unsupervised
* Supervised
    * Target(what model is predicting) is provided
    * Labeled data
    * Classification & regression
* Unsupervised
    * Target is unknow or UNAVAILABLE
    * Unlabeled data
    * Cluster analysis & association analysis
* Terms:
    * **Sample, record, example, row, instance, observation**
    * **Variable, attribute, field, feature, column, dimension**

#### scikit-learn
* Open source lib for ML in Python
* build on top of Numpy, SciPy, matplotlib
* Active community for development
* Improved continuously by developers

* Utility funcs for
    * Transforming raw feature vectors to suitable format
* API
    * Scaling of features: remove mean and keep unit variance
    * Normalization
    * Binarization 
    * One Hot encoding for categorical features
    * Handling of missing values
    * Generating higher order features
    * Build custom transformations
### Classification:
* binary-classification
* muti-class classification : what product will customer buy

* kNN, Decision Tree, Naive Bayes
* Decision Tree: split data into "pure" regions
    * Root Node
    * Internal Nodes
    * Leaf Nodes: when reach, determines the classification decision
* When to stop split:
    * All samples have same class label
    * Number of samples in node reaches min
    * Change in impurity measure is smaller than threshold 
    * Max tree depth is reached
#### Decision Trees demo
* Data Cleaning
    ```python
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
    ```  
* Convert to a Classification Task 
```python
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
# Fit on Train Set
humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
humidity_classifier.fit(X_train, y_train)
#Predict
predictions = humidity_classifier.predict(X_test)
#true data
y_test['high_humidity_label'][:10]
```
summary: 
```python
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
```
### Clustering
* Divides data into clusters
* Similar items are placed in same cluster
* Unsupervised, no 'correct' clustering, don't come with labels
####K-Means
* select k initial centroids
    * **k** select K initial centroids
    * **Repeat**
        * assign each sample in a data set to closest centroid  
        * Calculate mean of cluster to determine new centroid
    * Stopping criteria is reached
        * No changes to centroids
        * Number of samples changing clusters < threshold

* Issue: Final clusters are sensitive to init centroids
* Solution: Run K-means multiple times with diff init centroids

### Regression Analysis
Regression or Clustering
* when the model has to predict a **numeric value** instead of category
* Input Variables -> Model -> output (number)

* Training Data: Adjust model parameters
* Validation Data: 
    * Determine when to stop training (avoid overfitting)
    * Estimate generalization performance
* Test Data: Evaluate performance on new data

####Linear Regression
* Find regressio nline that makes sum of residuals as small as possible

