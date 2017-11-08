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
filter = (an_array > 15)
print(filter)