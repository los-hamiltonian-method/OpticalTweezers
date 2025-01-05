'''Circle detection algorithm.'''
#%% Imports
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

ndarray = np.ndarray
def detect_edges(img: ndarray, threshold: int, 
	center: Optional[Tuple[int, int]] = None,
	crop_D: Optional[Tuple[int, int]] = None,
	show=False) -> Tuple[ndarray, ndarray]:
	'''
	Detects edges on image array by converting image to binary map and then
	applying and edge detection kernel.
	If shown, use keys to destroy windows.
	Parameters
	----------
	img: ndarray
		Image array.
	center: Optional[Tuple[int, int]]
		(y, x) position from which to crop image.
	crop_D: Optional[Tuple[int, int]]
		(y, x) distance from center from which to crop image.
		If None, let's image as is.
	threshold: int
		Minimum value for which gray scale image pixels are converted
		to white on binary map.
	show: bool
		Show binary map and identified edges.

	Returns
	-------
	img_bw: ndarray
		ndarray of black and white image. Same dimensions as
		cropped image.
	edges: ndarray
		ndarray containing edges of cropped image. Same dimensions as
		cropped image.
	'''
	if center == None:
		# For an odd image, center is a 1x1 square.
		# For an even image, center is a 2x2 square.
		# For a combination, it is a 2x1 or 1x2 rectangle.
		# I consider the center to be the upper left corner
		# of this rectangle.
		shape = np.array(img.shape) - 1
		center = shape // 2

	if crop_D != None:
		crop_D = np.array(crop_D)
		img = img[center[0] - crop_D[0]:center[0] + crop_D[0] + 1,
	  	          center[1] - crop_D[1]:center[1] + crop_D[1] + 1]

    # Converting image to binary map with threshold.
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# TODO: Is there any gain in letting configure 255 upper bound?
	# I think yes.
	# TODO: Colored image support. Take differences in pixels.
	img_bw = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)[1]

	y_kernel = np.array([
		[-1, -2, -1],
		[0, 0, 0],
		[1, 2, 1]
	])
	x_kernel = np.transpose(y_kernel)
	y_deriv = cv.filter2D(src=img_bw, ddepth=-1, kernel=y_kernel)
	x_deriv = cv.filter2D(src=img_bw, ddepth=-1, kernel=x_kernel)
	inv_y_deriv = cv.filter2D(src=img_bw, ddepth=-1, kernel=-y_kernel)
	inv_x_deriv = cv.filter2D(src=img_bw, ddepth=-1, kernel=-x_kernel)
	edges = np.sqrt(x_deriv**2 + y_deriv**2
					+ inv_y_deriv**2 + inv_x_deriv**2).astype(float)
	edges = 255 * edges / np.max(edges)
	if show:
		cv.imshow('Binary map', img_bw)
		cv.imshow('Detected edges', edges)
		cv.waitKey(0)
		cv.destroyAllWindows()
	return img_bw, edges

def build_kernel(radius: int = 100, padding: int = 5,
	show=False) -> ndarray:
	'''Builds a single disk kernel array.'''
	min_R = radius - padding
	max_R = radius + padding

	# Substracting max_R as it is position of circles center on img array.
	Filter = lambda i, j: min_R < np.sqrt((i - max_R)**2
						 + (j - max_R)**2) < max_R
	kernel = [[Filter(i, j) for i in range(2 * max_R)]
			  for j in range(2 * max_R)]
	kernel = np.array(kernel).astype(float)
	# Weighting kernel based on center circle.
	kernel = kernel / (2 * np.pi * radius)

	if show:
		cv.imshow('kernel', kernel)
		cv.waitKey(0)
		cv.destroyAllWindows()
	
	return kernel

def filter_center(edges: ndarray, kernel: ndarray, threshold: int = None,
	show=False) -> Tuple[ndarray, ndarray]:
	'''Extracts circle center based on passes from disk kernel.'''

	center = cv.filter2D(src=edges, ddepth=-1, kernel=kernel)
	center_bw = cv.threshold(center, threshold, 255,
							    cv.THRESH_BINARY)[1]
	if show:
		cv.imshow('filtered', center)
		cv.imshow('bit filtered', center_bw)
		cv.waitKey(0)
		cv.destroyAllWindows()
	return center, center_bw

def argmax2d(array: ndarray) -> Tuple[int, int]:
	'''Returns (y, x) positions of maximum value in array.'''
	amax = array.argmax()
	x_shape = array.shape[1]
	row = int(np.floor(amax / x_shape))
	column = int((amax + 1) - (x_shape * row))
	return (row, column)

# TODO: I'm not using threshold2 nor center_bw as of now.
# TODO: Change calls to Optional[ndarray[int, int]]
# TODO: (y, x) for numpy, (x, y) for cv is a bit confusing.
def detect_circle(img: ndarray, radius: int, threshold1: int,
	kernel=ndarray, return_all: bool = False,
	threshold2: int = None, img_center: Optional[Tuple[int, int]] = None,
	crop_D: Optional[Tuple[int, int]] = None, show=False) -> Tuple[int, int]:
	'''Returns (x, y) coordinates of circle on img array.'''
	if crop_D and not img_center:
		img_center = tuple(((np.array(img.shape) - 1) // 2)[:-1])
	img_bw, edges = detect_edges(img, center=img_center, crop_D=crop_D,
								   threshold=threshold1)
	center_filter = filter_center(edges, kernel)[0]
	center = argmax2d(center_filter)
	if crop_D:
		center = np.array(img_center) - np.array(crop_D) + center
	center = center[::-1]
	img = cv.circle(img, center, radius=0, color=(0, 0, 255),
					thickness=3)
	img = cv.circle(img, center, radius=radius, color=(0, 0, 255),
					thickness=1)
	if show:
		show_circle(img, center, radius, img_bw,
					edges, kernel, center_filter)
	if return_all:
		return center, img, img_bw, edges, center_filter
	return center, img

def show_circle(img, center, radius, img_bw, edges, kernel, center_filter):
	plt.style.use('dark_background')
	# Adding circle to image
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

	# Mosaic
	mosaic = [['img_bw', 'edges', 'img', 'img'],
			  ['kernel', 'center_filter', 'img', 'img']]
	keys = ['img_bw', 'edges', 'kernel', 'center_filter', 'img']
	titles = ['Binary map', 'Edges', 'Kernel', 'Filtered center',
			  'Original Image']
	imgs = [img_bw, edges, kernel, center_filter, img]

	# Plots
	fig, ax = plt.subplot_mosaic(mosaic)
	fig.suptitle('Circle Identification')
	fig.patch.set_facecolor('#787878')
	for key, im, title in zip(keys, imgs, titles):
		ax[key].imshow(im)
		ax[key].set(title=title)

	# Show
	plt.subplots_adjust(hspace=0, wspace=0.35)
	#figManager = plt.get_current_fig_manager()
	#figManager.full_screen_toggle()
	plt.show()


#%% main()
def main():
	import os
	import sys
	sys.path.insert(0, '../MiniPys/Formatter')
	import minipy_formatter as MF
	MF.Format().rcUpdate()
	# Images
	# img_name = 'micro.png'
	# img_name = 'micro2.bmp'
	# img_name = 'eye.webp'
	# radius
	# micro: 42, 1, 81
	# micro 2: 67, 1, 78
	# iris: 150, 1, 140
	# pupil: 61, 1, 40

	img_dir = './Circle detection/images/'
	img_name = 'micro.png'
	img = cv.imread(os.path.join(img_dir, img_name))
	
	# Kernel params
	radius = 42
	padding = 1
	kernel = build_kernel(radius=radius, padding=padding)

	# Params
	threshold = 81
	img_center = None
	crop_D = None

	detect_circle(img, radius=radius, kernel=kernel, show=True,
	     		  threshold1=threshold, img_center=img_center, crop_D=crop_D)


#%% 呪い
if __name__ == '__main__':
	main()
	