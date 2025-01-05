import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from typing import Tuple

def argmax2d(array: ndarray) -> Tuple[int, int]:
	'''Returns (y, x) positions of maximum value in array.'''
	amax = array.argmax() / 3
	x_shape = array.shape[1]
	row = int(np.floor(amax / x_shape))
	column = int(amax - (x_shape * row))
	return (row, column)

h = 255 / 2
A = np.array([
	[h, h, h, h, h, h],
	[h, h, h, h, h, 255],
	[h, h, h, h, h, h],
	[h, h, h, h, h, h],
	[h, h, h, h, h, h],
	[h, h, h, h, h, h],
])
A = cv.imread('./filter.png')
Max = argmax2d(A)
print(A.argmax())
print(A.shape)
print(A)
A[Max] = 0
print(A.argmax())
print(Max)
A = cv.circle(A, Max[::-1], radius=1, thickness=3, color=(0, 255, 0))
plt.imshow(A)
plt.show()
