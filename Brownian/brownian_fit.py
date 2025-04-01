import matplotlib.pyplot as plt
import numpy as np
import json, sys, os
import pandas as pd
from pprint import pprint
from scipy import optimize
from scipy.stats import norm
from uncertainties import ufloat


def main():
	formatter_path = r"..\..\MiniPys\Formatter"
	sys.path.insert(0, formatter_path)
	import minipy_formatter as MF
	plt.style.use('default')
	MF.Format().rcUpdate()

	dirpath = '../Data/20-01-25/Brownian analysis'
	brownian_fitpath = os.path.join(dirpath, 'brownian_fit.csv')
	df = pd.read_csv(brownian_fitpath)
	current = df['current(mA)']
	kx = df['kx']
	covx = df['covx']
	ky = df['ky']
	covy = df['covy']
	
	fig, axs = plt.subplot_mosaic([['kx'], ['ky']], dpi=200, figsize=(20, 10))
	axs['kx'].scatter(current, kx)
	axs['ky'].scatter(current, ky)

	fig.suptitle('Brownian Fits')
	xlabels = ["$k_x$ [kg / s$^2$]", "$k_y$ [kg / s$^2$]"]
	for ax, label in zip(axs, xlabels):
		axs[ax].set(title=label, xlabel='Current [mA]', ylabel=label)
		axs[ax].grid(True, ls='--')

	plt.subplots_adjust(hspace=0.5, wspace=0.3)
	plt.savefig(os.path.join(dirpath, 'brownian_fit.png'))


if __name__ == '__main__':
	main()