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

	dirpath = '../Data/20-01-25/MSD analysis'
	MSD_fitpath = os.path.join(dirpath, 'MSD_fit.csv')
	df = pd.read_csv(MSD_fitpath)
	df = pd.read_csv(MSD_fitpath)
	current = df['current(mA)']
	tau0x = df['tau0x']
	Ax = df['Ax']
	tau0y = df['tau0y']
	Ay = df['Ay']
	
	fig, axs = plt.subplot_mosaic([['tau0x', 'tau0y'], ['Ax', 'Ay'], ['kTx', 'kTy']],
								  dpi=200, figsize=(20, 10))
	axs['tau0x'].scatter(current, 1 / tau0x)
	axs['Ax'].scatter(current, 1 / Ax)
	axs['tau0y'].scatter(current, 1 / tau0y)
	axs['Ay'].scatter(current, 1 / Ay)

	axs['kTx'].scatter(current, Ax / tau0x, label='From $x$-data')
	axs['kTy'].scatter(current, Ay / tau0y, label='From $y$-data')

	fig.suptitle('MSD Fits')
	xlabels = ["$k_x / \\gamma$ [s$^{-1}$]", "$k_y / \\gamma$ [s$^{-1}$]",
			   "$k_x / 2 k_B T$ [nm$^{-2}$]", "$k_y / 2 k_B T$ [nm$^{-2}$]",
			   "$2 k_B T / \\gamma$ ($x$-data)", "$2 k_B T / \\gamma$ ($y$-data)"]
	for ax, label in zip(axs, xlabels):
		axs[ax].set(title=label, xlabel='Current [mA]', ylabel=label)
		axs[ax].grid(True, ls='--')

	plt.subplots_adjust(hspace=0.5, wspace=0.3)
	plt.savefig(os.path.join(dirpath, 'MSD_fit.png'))


if __name__ == '__main__':
	main()