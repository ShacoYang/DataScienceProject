import numpy as np

an_array = np.array([[11,12,13,14],[21,22,23,24],[31,32,33,34]])
print (an_array)

#use array slicing to get a subarray consisting of the first 2 rows * 2 columns
'''
a_slice has its own indices
there indices are different than the indices in an_array
'''
a_slice = an_array[:2 , 1:3]
print (a_slice)
'''
[[12 13]
 [22 23]]
'''

# slices are just a references to the same underlying data as the original array
print ("before:", an_array[0,1])
a_slice[0,0] = 1000 #same as an_array[0,1]
print ("after:", an_array[0,1])

a_slice = np.array(an_array[:2, 1:3]) #create a copy of the portion of np.array

# using the SINGLE INDEX is a SPECIAL CASE
an_array = np.array([[11,12,13,14],[21,22,23,24],[31,32,33,34]])
row_rank1 = an_array[1, :]
#only a single []
print (row_rank1, row_rank1.shape)
'''(array([21, 22, 23, 24]), (4,))'''
row_rank2 = an_array[1:2, :]
print (row_rank2, row_rank2.shape)

# same thing for columns of an array
col_rank1 = an_array[:, 1]
col_rank2 = an_array[:, 1:2]
print (col_rank1, col_rank1.shape)
print (col_rank2, col_rank2.shape)


#Array indexing for changing elements
an_array = np.array([[11,12,13],[21,22,23],[31,32,33],[41,42,43]])
print('original array:')
print (an_array)

#create an array of indices
col_indices = np.array([0,1,2,0])
print('\nCol indices picked:', col_indices)
row_indices = np.arange(4)
print('\nRows indices pickedL', row_indices)

#pairing of row and col
for row, col in zip(row_indices, col_indices):
    print (row, ", " , col)
#select one from each row
print("Values in the array at those indices: ", an_array[row_indices, col_indices])

an_array[row_indices, col_indices] += 10000
print ('\nChanged Array:')
print (an_array)

# create 3 * 2 array
an_array = np.array([[11,12],[21,22],[31,32]])
print(an_array)
#create a filter which will be boolean values
filter = (an_array > 15)
print(filter)
#select elements which meet the criteria
print(an_array[filter])
#for short,
an_array[(an_array > 20) & (an_array < 30)]

#change the element based on condition
an_array[an_array % 2 == 0] += 100
print(an_array)


#Datatypes:
ex1 = np.array([11,12])
print(ex1.dtype)
print(type(ex1))

ex2 = np.array([11.0,12.0])
print(ex2.dtype)

ex3 = np.array([11,21], dtype = np.int64)
print(ex3.dtype)

ex4 = np.array([11.1,12.7], dtype=np.int64)
print(ex4.dtype)
print(ex4)

#Arithmetic Operations:
x = np.array([[111,112],[121,122]], dtype=np.int64)
y = np.array([[211.1,212.2],[221.1,222.1]], dtype=np.float64)

print(x)
print(y)

print(x + y)
#same as the numpy function "add"
print(np.add(x, y))

print(x - y)
print(np.subtract(x,y))

# Statistical, Sorting, Set operations
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

#Sorting
unsorted = np.random.randn(5)
print(unsorted)
# create copy and sort
sorted = np.array(unsorted)
sorted.sort()
print(sorted)
print(unsorted)
#inplace sorting
unsorted.sort()
print(unsorted)

#Finding Unique elements:
array  = np.array([1,2,3,4,1,2,4,2])
print(np.unique(array))

#Set Operations with np.array data type
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

#Broadcasting
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
#broadcast in both dimensions
add_scalar = np.array([1])
print(start + add_scalar)
# b1 b2
a = np.array([[0,0],[0,0]])
b1 = np.array([1,1])
b2 = 1
print(a + b1 == a + b2)
print(a + b2)

#SpeedTest: ndarrays VS lists
from numpy import arange
from timeit import Timer
size = 1000000
timeits = 1000

nd_array = arange(size)
print(type(nd_array))
#timer expects the operation as a parameter,
#pass nd_array.sum()
timer_numpy = Timer("nd_array.sum()", "from __main__ import nd_array")
print("Time taken by numpy ndarray: %f seconds" %
      (timer_numpy.timeit(timeits)/timeits))
a_list = list(range(size))
print(type(a_list))
timer_list = Timer("sum(a_list)", "from __main__ import a_list")
print("Time taken by list: %f seconds" %
      (timer_list.timeit(timeits)/timeits))