# calculating euclidean distance between vectors
from math import sqrt
 
# calculate euclidean distance
def euclidean_distance(a, b):
    return sqrt(sum((e1-e2)**2 for e1, e2 in zip(a,b)))
 
# define data
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]
# calculate distance
dist = euclidean_distance(row1, row2)
print("Euclidean distance : ")
print(dist)