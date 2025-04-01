import matplotlib.pyplot as plt
import numpy as np
import json, sys, os
import pandas as pd
from pprint import pprint
from scipy import optimize
from scipy.stats import norm
from uncertainties import ufloat


def load_data(video_name, data):
	global time_step, current, xposition, yposition
	time_step = data[video_name]['time_step']
	current = video_name.split('_')[0]
	current = float(current.replace('-', '.').replace('mA', ''))
	positions = np.array(data[video_name]['centers'])
	xposition = resolution.n * positions[:,0,0]
	yposition = resolution.n * positions[:,0,1]


def MSD(position):
	MSD = [0]
	for M in range(1, len(position)):
		msd = (position[M:] - position[:-M])**2
		msd = sum(msd) / len(position[M:])
		MSD.append(msd)
	return np.array(MSD)


def analytic_MSD(tau, tau_0, A):
	return A * (1 - np.exp(-np.abs(tau) / tau_0))


def save_popt(xpopt, ypopt):
	df_dict['current(mA)'].append(current)
	df_dict['tau0x'].append(xpopt[0])
	df_dict['Ax'].append(xpopt[1])
	df_dict['tau0y'].append(ypopt[0])
	df_dict['Ay'].append(ypopt[1])


def plot_MSD(show=False):
	fig, axs = plt.subplot_mosaic([['xpos'], ['ypos']], dpi=200, figsize=(20, 10))
	xmsd, ymsd = MSD(xposition), MSD(yposition)
	time = time_step * np.array(range(len(xposition)))

	# Slicing
	time_filter = time < cutoff_time
	xmsd = xmsd[time_filter]
	ymsd = ymsd[time_filter]
	time = time[time_filter]

	# Fit
	bounds = ((1E-3, 0), (np.inf, np.inf)) # tau_0, A
	xpopt, xpcov = optimize.curve_fit(analytic_MSD, time, xmsd, bounds=bounds)
	ypopt, ypcov = optimize.curve_fit(analytic_MSD, time, ymsd, bounds=bounds)

	# Plots
	axs['xpos'].scatter(time, xmsd, label='MSD$_x$ data', s=5)
	axs['xpos'].plot(time, analytic_MSD(time, *xpopt), label='MSD$_x$ fit') 
	axs['ypos'].scatter(time, ymsd, label='MSD$_y$ data', s=5)
	axs['ypos'].plot(time, analytic_MSD(time, *ypopt), label='MSD$_y$ fit')

	# v/hlines
	axs['xpos'].axvline(xpopt[0], color='#ccc', ls='-.', label='$\\tau_{0,x}$')
	axs['ypos'].axvline(ypopt[0], color='#ccc', ls='-.', label='$\\tau_{0,y}$')
	axs['xpos'].axhline(xpopt[1], color='#aaa', ls='--', label='$2 k_B T / k_x$')
	axs['ypos'].axhline(ypopt[1], color='#aaa', ls='--', label='$2 k_B T / k_y$')

	# Format
	for ax in axs: axs[ax].legend()
	fig.suptitle(video_name)
	axs['xpos'].set(title="MSD$_x$", ylabel="MSD$_x$($\\tau$) [nm$^2$]",
					xlabel="$\\tau$ [s]")
	axs['ypos'].set(title="MSD$_y$", ylabel="MSD$_y$($\\tau$) [nm$^2$]",
		            xlabel="$\\tau$ [s]")
	plt.subplots_adjust(hspace=0.5)

	# I/O
	figpath = os.path.join(dir_path, 'MSD analysis', 'Figures',
						   'MSD_' + video_name.replace('.avi', '.png'))
	plt.savefig(figpath)
	save_popt(xpopt, ypopt)

	if show: plt.show()


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

	global video_name, cutoff_time, df_dict
	df_dict = {'current(mA)': [], 'tau0x': [], 'Ax': [], 'tau0y': [], 'Ay': []}
	cutoff_time = 0.5 # s
	for video_name in data.keys():
		if video_name in skip: continue
		load_data(video_name, data)
		plot_MSD()
		print(video_name)

	df_fit = pd.DataFrame.from_dict(df_dict)
	df_path = os.path.join(dir_path, 'MSD analysis', 'MSD_fit.csv')
	df_fit.to_csv(df_path)


if __name__ == '__main__':
	main()