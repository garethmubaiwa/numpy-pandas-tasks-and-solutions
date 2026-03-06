import numpy as np

rng = np.random.default_rng(seed = 1)
import random
import math

# Given a random array, change the sign of elements whose values ​​are between 3 and 8
a1 = rng.integers(1, 20, 10)
mask = (3<a1) & (a1<14)
a1 = np.where(mask , -a1, a1)
print('Array after changing signs:',a1)

# Replace the maximum element of a random array with 0
a2 = rng.integers(0, 10, 10, endpoint=True)
mask1 = np.argmax(a2)
print('Original array:',a2)
a2[mask1] = 0
print('Index of max element:', mask1)
print('Max elements with zero:',a2)

# Construct a direct product of arrays (all combinations with each element). The input is a two-dimensional array
x = rng.integers(0, 10, size= (2,2))
y = rng.integers(0, 10, size=(3,3))
xy = np.array([np.concatenate((X,Y)) for X in x for Y in y]) #meshgrid and stack for larger arrays
print('Direct product of arrays:', xy)

# Given two arrays, A (8x3) and B (2x2), find rows in A that contain elements from each row in B, regardless of the order of the elements in B.
A = rng.integers(0, 10, size=(8,3))
B = rng.integers(0, 10, size=(2,2))

print('A:\n', A)
print('B:\n', B)

match0 = np.isin(A, B[0]).any(axis=1)
match1 = np.isin(A, B[1]).any(axis=1)

matched = np.where((match0 & match1))[0]
print('row indices:\n', matched)
row = A[matched]
print('row values: \n', row)

# Given a 10x3 matrix, find rows of unequal values ​​(for example, row [2,2,3] remains, row [3,3,3] is removed)
A = rng.integers(0,10, size=(10,3))
print('Original matrix:\n',A)
mask = np.any(A != A[:, [0]], axis=1) #differences
#mask = ~(A == A[:, [0]]).all(axis=1) #equal value
print('Changed matrix:\n',A[mask])

# Given a two-dimensional array, remove the rows that are repeated.
a = np.array([[1,2,3],[3,5,6],[0,1,3],[8,4,6],[1,2,3],[0,1,3],[4,7,2],[8,1,0],[9,3,9],[9,9,9]])
print('a: \n',a)
unique_row= np.unique(a, axis=0)
print('Unique_row: \n', unique_row)

''' For each of the following problems (1-5), you need to provide two implementations - one without using numpy (assume that where the input or output should be numpy arrays, there will be just lists), 
# and the second fully vectorized one using numpy (without using Python loops/map/list comprehension).
# Note 1. You can assume that all the specified objects are non-empty (for example, in problem 1, there are non-zero elements on the diagonal of the matrix). 
# Note 2. For most problems, the solution takes no more than 1-2 lines.'''

# Problem 1: Calculate the product of nonzero elements on the diagonal of a rectangular matrix. For example, for X = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]]) the answer is 3
x = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]])
#without numpy
row = len(x)
col = len(x[0])
diag_len = min(row, col)
diag = [x[i][i] for i in range(diag_len) if x[i][i] != 0]
diag_product = math.prod(diag)
print('Product:',diag_product)
#numpy
diag_np = np.diag(x) 
product = np.prod(diag_np[diag_np != 0])
print('Product (numpy):',product)

# Problem 2: Given two vectors x and y, check whether they define the same multiset. For example, for x = np.array([1, 2, 2, 4]), y = np.array([4, 2, 1, 2]) the answer is True.
x = np.array([1, 2, 2, 4])
y = np.array([4, 2, 1, 2])
#without numpy
from collections import Counter

same = Counter(x) == Counter(y)
print('Same multiset:',same)

#numpy
same_np = np.array_equal(np.sort(x), np.sort(y))
print('Same multiset (numpy):',same_np)

# Problem 3: Find the maximum element in the vector x among elements preceded by zero. For example, for x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0]) the answer is 5
x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])
#without numpy

target = [x[i+1] for i in range(len(x)-1) if x[i] == 0]
print('Max element:',max(target))

#numpy
zero_mask = x[:-1]==0
target1 = x[1:][zero_mask]
print('Max element (numpy):',np.max(target1))

# Problem 4: Implement run-length encoding. For some vector x, return a tuple of two vectors of the same length. 
#The first contains numbers, and the second contains the number of times they should be repeated. For example, for x = np.array([2, 2, 2, 3, 3, 3, 5]), the answer is (np.array([2, 3, 5]), np.array([3, 3, 1])).
#numpy
x = np.array([2, 2, 2, 3, 3, 3, 5])
idx = np.concatenate(([0], np.where(np.diff(x) != 0)[0] + 1, [len(x)]))
values, counts = x[idx[:-1]], np.diff(idx)
print(np.array([values]), np.array([counts]))

#without numpy
from itertools import groupby
values, counts = zip(*[(k, sum(1 for _ in g)) for k, g in groupby(x)])
print(np.array([values]), np.array([counts]))

# Problem 5: Given two samples of objects X and Y, calculate the matrix of Euclidean distances between the objects. Compare this with the scipy.spatial.distance.cdist function in terms of speed.
X = np.array([[0, 0], [1, 0], [2, 0]])
Y = np.array([[0, 1], [1, 1]])
#numpy
diff = X[:, None, :] - Y[None, :, :]
d = np.linalg.norm(diff, axis=2)
print('Euclidean Distance (numpy):\n',d)

#scipy
from scipy.spatial.distance import cdist
dist_matrix = cdist(X, Y, metric='euclidean')
print('Euclidean Distance (scipy):\n',dist_matrix)

'''Problem 6: CrunchieMunchies * 
- You work in the marketing department of MyCrunch, a food company that is developing a new type of delicious, healthy cereal called CrunchieMunchies. 
- You want to demonstrate to consumers how healthy your cereal is compared to other leading brands, so you have collected nutrition data from several different competitors. 
- Your task is to use Numpy calculations to analyze this data and prove that your Crunchie Munchies are the healthiest choice for consumers.'''

# View the cereal.csv file. This file contains calorie counts for various brands of cereal. Download the data from the file and save it as calorie_stats.
calorie_stats = np.loadtxt("cereal.csv", delimiter=",")
print('Calorie stats:\n', calorie_stats)

# 1. One serving of CrunchieMunchies contains 60 calories. How much higher is the average calorie count of your competitors?
average_calories = np.mean(calorie_stats) - 60
print('Average calories difference:', average_calories)

# 2. Sort the data and store the result in the variable calorie_stats_sorted. Print the sorted information.
calorie_stats_sorted = np.sort(calorie_stats)
print('Sorted calorie stats:\n', calorie_stats_sorted)

# 3. Calculate the median of the data set and store your answer in median_calories. Print the median so you can see how it compares to the mean.
median_calories = np.median(calorie_stats)
print('median_calories:', median_calories)

# 4. Calculate various percentiles and print them until you find the lowest percentile that exceeds 60 calories. Store this value in the nth_percentile variable and print it.
percentiles = np.percentile(calorie_stats, np.arange(101))
nth_percentile = percentiles[percentiles > 60][0]
print('nth_percentile:', nth_percentile)

# 5. Instead, let's calculate the percentage of cereals that contain over 60 calories per serving. Store your answer in the more_calories variable and print it.
more_calories = np.mean(calorie_stats > 60) * 100
print('Percentage of cereals with more than 60 calories:', more_calories)

# 6. Calculate the magnitude of the deviation by finding the standard deviation. Save your answer in calorie_std and print it to the terminal. How can we incorporate this value into our analysis?
calorie_std = np.std(calorie_stats)
print('Standard deviation of calorie counts:', calorie_std)

# 7. Write a short paragraph outlining your findings and how you believe this data can be used to Mycrunch's advantage when marketing CrunchieMunchies.
'''The average calorie count of competitors is significantly higher than that of CrunchieMunchies. The mean difference of about 47 calories, median of 110 calories and over 96% of competitor cereals contain more 60 calories per serving. The standard deviation indicates moderate variation between competitors in calorie count, the lowest percentile greater than 60 is 70, indicating that almost all cereals have higher calorie count than CruchieMunchies.
CrunchieMunchies stands as a low-calorie alternative in a market dominated by high calorie cereals.
Can be marketed as a healthier cereal option for calorie conscious consumers'''