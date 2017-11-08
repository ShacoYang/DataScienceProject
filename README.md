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



