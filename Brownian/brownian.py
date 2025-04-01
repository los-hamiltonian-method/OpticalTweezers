import matplotlib.pyplot as plt
import numpy as np
import json, sys, os
from pprint import pprint
from scipy import optimize
from scipy.stats import norm
import pandas as pd
from uncertainties import ufloat


def get_positions(video_data):
	centers = np.array(video_data['centers'])[:,0,:]
	num_centers = len(centers)
	
	global times, time_step, x_centers, y_centers
	time_step = video_data['time_step']
	times = np.array([i * time_step for i in range(0, num_centers)])
	x_centers = resolution.n * centers[:, 0].astype('float64')
	y_centers = resolution.n * centers[:, 1].astype('float64')

	# Shifting
	x_centers -= x_centers.mean()
	y_centers -= y_centers.mean()

def save_positions(show=False):
	fig, axs = plt.subplot_mosaic([['x'], ['y']], dpi=200)
	labels = ['x', 'y']
	axs['x'].plot(times, x_centers, linewidth=0.5)
	axs['y'].plot(times, y_centers, color='red', linewidth=0.5)

	fig.suptitle(f"Brownian motion for {video_name}")
	for c, label in zip((x_centers, y_centers), axs):
	    axs[label].set(xlabel=f'Time (t), step={round(time_step, 4)}',
	                   ylabel=f'${label}$-displacement (nm)')
	    axs[label].axhline(0, color='#888', linestyle='--', linewidth=1)
	    axs[label].axhline(np.max(c), color='#888',
	    	               linestyle='--', linewidth=1)
	    axs[label].axhline(np.min(c), color='#888',
	    	               linestyle='--', linewidth=1)

	figpath = os.path.join(dir_path, 'Brownian analysis/Figures',
				f"position_{video_name.replace('.avi', '.png')}")
	plt.savefig(figpath)
	if show: plt.show()


ndarray = np.ndarray
def get_variance(distribution: ndarray, values: ndarray, mean: float = 0):
	variance = 0
	for p, x in zip(distribution, values): 
		variance += p * x**2
	return variance - mean**2

# TODO: Save figs.
def get_position_distribution(show=False):
	# Ranges
	binwidth = 4
	X = np.linspace(min(x_centers), max(x_centers), 3000)
	Y = np.linspace(min(y_centers), max(y_centers), 3000)
	x_binrange = np.arange(min(x_centers), max(x_centers) + binwidth, binwidth)
	y_binrange = np.arange(min(y_centers), max(y_centers) + binwidth, binwidth)
	
	# Plots
	fig, axs = plt.subplot_mosaic([['x'], ['y']], dpi=200)
	global hist_x, hist_y
	hist_x = axs['x'].hist(x_centers, bins=y_binrange, density=False, alpha=0.4)
	hist_y = axs['y'].hist(y_centers, bins=y_binrange, density=False,
		                   color='red', alpha=0.4)

	# Gaussian fits
	x_std = np.std(x_centers)
	y_std = np.std(y_centers)
	axs['x'].plot(X, norm.pdf(X, 0, x_std), color='blue',
		          label=f"$\sigma$ = {x_std:.2f}")
	axs['y'].plot(Y, norm.pdf(Y, 0, y_std), color='red',
				  label=f"$\sigma$ = {y_std:.2f}")

	# Format
	fig.suptitle(f"Position distribution for {video_name}")
	for label, std in zip(axs, (x_std, y_std)):
	    axs[label].set(xlabel=f'Position [nm]',
	                   ylabel=f'Counts')
	    axs[label].legend(loc='upper right')

	figpath = os.path.join(dir_path, 'Brownian analysis/Figures',
				f"distribution_{video_name.replace('.avi', '.png')}")
	plt.savefig(figpath)
	if show: plt.show()

	return fig, axs

def get_potential(show=False):
	x = -np.log(hist_x[0])
	y = -np.log(hist_y[0])
	x = x - min(x)
	y = y - min(y)
	bin_x = (hist_x[1][:-1] + hist_x[1][1:]) / 2
	bin_y = (hist_y[1][:-1] + hist_y[1][1:]) / 2
	bin_x, bin_y = bin_x[x < np.inf], bin_y[y < np.inf]
	x, y = x[x < np.inf], y[y < np.inf]
	fig, axs = plt.subplot_mosaic([['x'], ['y']], dpi=200)
	axs['x'].scatter(bin_x, x, color='blue', s=3)
	axs['y'].scatter(bin_y, y, color='red', s=3)

	# Fit
	f = lambda r, a: a * r**2
	X = np.linspace(min(x_centers), max(x_centers), 3000)
	Y = np.linspace(min(y_centers), max(y_centers), 3000)
	xopt, xcov = optimize.curve_fit(f, bin_x, x)
	yopt, ycov = optimize.curve_fit(f, bin_y, y)
	axs['x'].plot(X, f(X, xopt[0]), color='blue',
		label=f"Fit: $a x^2$\n$a={float(xopt[0]):.2E}\\pm{float(xcov[0]):.1E}$")
	axs['y'].plot(Y, f(Y, yopt[0]), color='red',
		label=f"Fit: $a x^2$\n$a={float(yopt[0]):.2E}\\pm{float(ycov[0]):.1E}$")

	# Format
	fig.suptitle(f"Potenatials for {video_name}")
	for label in axs:
	    axs[label].set(xlabel=f'Position [nm]',
	                   ylabel=f'Potential [kT]')
	    axs[label].legend(loc='upper right')
	figpath = os.path.join(dir_path, 'Brownian analysis/Figures',
		f"potential_{video_name.replace('.avi', '.png')}")
	plt.savefig(figpath)
	if show: plt.show()

	update_fit(xopt, xcov, yopt, ycov)
	return fig, axs


def update_fit(xopt, xcov, yopt, ycov):
	df_dict['current(mA)'].append(current)
	df_dict['kx'].append(2 * xopt[0])
	df_dict['covx'].append(2 * xcov[0])
	df_dict['ky'].append(2 * yopt[0])
	df_dict['covy'].append(2 * ycov[0])

def main():
	formatter_path = r"..\..\MiniPys\Formatter"
	sys.path.insert(0, formatter_path)
	import minipy_formatter as MF
	plt.style.use('default')
	MF.Format().rcUpdate()

	# Resolution
	resolution_file = '../Data/20-01-25/resolution.csv'
	global resolution
	resolution = pd.read_csv(resolution_file)['resolution(um/px)'][0] # um / px
	resolution = resolution.split('+/-')
	resolution = ufloat(float(resolution[0]), float(resolution[1]))
	resolution = resolution * 1E3 # nm / px

	skip = {'63-00mA_1.avi', '63-50mA_3.avi', '63-75mA_2.avi',
			'64-50mA_2.avi', '65-00mA_2.avi','65-00mA_3.avi',
			'65-50mA_2.avi','66-00mA_2.avi','66-00mA_3.avi'}
	global dir_path	
	dir_path = '../Data/20-01-25/'
	json_path = os.path.join(dir_path, 'tracking.json')
	with open(json_path, 'r') as f:
	    data = json.load(f)['main']

	global video_name, df_dict, current
	df_dict = {'current(mA)': [], 'kx': [], 'covx': [], 'ky': [], 'covy': []}
	for video_name in data.keys():
		if video_name in skip: continue
		current = video_name.split('_')[0]
		current = float(current.replace('-', '.').replace('mA', ''))
		get_positions(data[video_name])
		save_positions()
		get_position_distribution()
		get_potential()

	df_fit = pd.DataFrame.from_dict(df_dict)
	df_path = os.path.join(dir_path, 'Brownian analysis', 'brownian_fit.csv')
	df_fit.to_csv(df_path)


if __name__ == '__main__':
	main()
