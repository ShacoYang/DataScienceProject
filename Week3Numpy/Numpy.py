import numpy as np
# create a rank 1 array
an_array = np.array([3,33,333])
print (type(an_array))

# test the shape of the array, it should have just one dimension
print (an_array.shape)

#sine it's a 1-rank array, we need only one index to access each element
print (an_array[0], an_array[1], an_array[2])

# ndarrays are mutable,
an_array[0] = 1234
print (an_array)


# rank 2 array
another = np.array([[11,12,13],[21,22,23]])
print (another)
print ("The shape is 2 rows, 3 columns:", another.shape) #rows x colums
print ("Accessing elements [0,0] [0,1] and [1,0] of the nparray:" , another[0,0], another[0,1], another[1,0])

# 2 * 2 array of zeros

ex1 = np.zeros((2,2))
print (ex1)
# 2 * 2 array of 9.0
ex2 = np.full((2,2), 9.0)
print (ex2)
# diagonal 1s and others 0
ex3 = np.eye(2,2)
print (ex3)
# array of ones
# one by 2 matrix not a rank one matrix
ex4 = np.ones((1,2))
print (ex4)
#rank2 it's 2 * 1
print(ex4.shape)
#need two indexes to access an element
print(ex4[0,1])
# create an array of random floats between 0 to 1
ex5 = np.random.random((2,2))
print (ex5)