import numpy as np
from numpy import array



last_centers = [array([109,  99]), array([102, 115]), array([127,  95]), array([109, 131]), array([140, 106]), array([126, 134]), array([140, 123]), array([149, 115]), array([164, 115]), array([178, 116]), array([191, 115]), array([204, 116]), array([217, 115]), array([213, 133])]
centers = [array([204, 116]), array([127,  95]), array([214, 133]), array([109, 131]), array([140, 106]), array([102, 115]), array([109,  99]), array([217, 116]), array([126, 135]), array([140, 123]), array([191, 115]), array([177, 115]), array([164, 115]), array([149, 115])]
n_circles = len(last_centers)

def distance_squared(array1, array2):
	'''Returnes square of separation distance between array1 and array2.'''
	x_2 = np.vectorize(lambda x: x**2)
	return sum(x_2(array1 - array2))

ordered_centers = n_circles * [None]
last_centers = list(enumerate(last_centers))
for center in centers:
	key = lambda lcenter: distance_squared(center, lcenter[1])
	last_centers.sort(key=key)
	ordered_centers[last_centers.pop(0)[0]] = center
centers = ordered_centers

print(centers)
