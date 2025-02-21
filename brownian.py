import matplotlib.pyplot as plt
import numpy as np
import json
import sys
from pprint import pprint
from scipy import optimize
from scipy.stats import norm


def get_positions(video_data):
	centers = np.array(video_data['centers'])[:,0,:]
	num_centers = len(centers)
	
	global times, time_step, x_centers, y_centers
	time_step = video_data['time_step']
	times = np.array([i * time_step for i in range(0, num_centers)])
	x_centers = centers[:, 0].astype('float64')
	y_centers = centers[:, 1].astype('float64')

	# Shifting
	x_centers -= x_centers.mean()
	y_centers -= y_centers.mean()

def show_positions():
	fig, axs = plt.subplot_mosaic([['x'], ['y']], dpi=200)
	labels = ['x', 'y']
	axs['x'].plot(times, x_centers, linewidth=0.5)
	axs['y'].plot(times, y_centers, color='red', linewidth=0.5)

	fig.suptitle(f"Brownian motion for {video_name}")
	for c, label in zip((x_centers, y_centers), axs):
	    axs[label].set(xlabel=f'Time (t), step={round(time_step, 4)}',
	                   ylabel=f'${label}$-displacement (px)')
	    axs[label].axhline(0, color='#888', linestyle='--', linewidth=1)
	    axs[label].axhline(np.max(c), color='#888',
	    	               linestyle='--', linewidth=1)
	    axs[label].axhline(np.min(c), color='#888',
	    	               linestyle='--', linewidth=1)

	plt.show()


ndarray = np.ndarray
def get_variance(distribution: ndarray, values: ndarray, mean: float = 0):
	variance = 0
	for p, x in zip(distribution, values): 
		variance += p * x**2
	return variance - mean**2

# TODO: Save figs.
def show_histogram():
	# Ranges
	binwidth = 0.2
	X = np.linspace(min(x_centers), max(x_centers), 3000)
	X = np.linspace(min(y_centers), max(y_centers), 3000)
	x_binrange = np.arange(min(x_centers), max(x_centers) + binwidth, binwidth)
	y_binrange = np.arange(min(y_centers), max(y_centers) + binwidth, binwidth)
	
	# Plots
	fig, axs = plt.subplot_mosaic([['x'], ['y']], dpi=200)
	hist_x = axs['x'].hist(x_centers, bins=y_binrange, density=True, alpha=0.4)
	hist_y = axs['y'].hist(y_centers, bins=y_binrange, density=True,
		                   color='red', alpha=0.4)

	# Gaussian fits
	x_variance = get_variance(hist_x[0], hist_x[1])
	y_variance = get_variance(hist_y[0], hist_y[1])
	axs['x'].plot(X, norm.pdf(X, 0, np.sqrt(x_variance)), color='blue',
		          label=f"$\sigma$ = {x_variance:.2f}")
	axs['y'].plot(X, norm.pdf(X, 0, np.sqrt(y_variance)), color='red',
				  label=f"$\sigma$ = {y_variance:.2f}")

	# Format
	fig.suptitle(f"Position distribution for {video_name}")
	for label, variance in zip(axs, (x_variance, y_variance)):
	    axs[label].set(xlabel=f'Position (px)',
	                   ylabel=f'Counts')
	    axs[label].legend(loc='upper right')
	plt.show()

	return fig, axs


def main():
	formatter_path = r"..\MiniPys\Formatter"
	sys.path.insert(0, formatter_path)
	import minipy_formatter as MF
	plt.style.use('default')
	MF.Format().rcUpdate()

	skip = {'63-00mA_1.avi', '63-50mA_3.avi', '63-75mA_2.avi', '64-50mA_2.avi', '65-00mA_2.avi','65-00mA_3.avi','65-50mA_2.avi','66-00mA_2.avi','66-00mA_3.avi'}
	json_path = './Images/20-01-25/tracking.json'
	with open(json_path, 'r') as f:
	    data = json.load(f)['main']

	global video_name
	for video_name in data.keys():
		if video_name in skip: continue
		get_positions(data[video_name])
		# show_positions()
		show_histogram()



if __name__ == '__main__':
	main()
