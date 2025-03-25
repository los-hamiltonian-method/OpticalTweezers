import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def main():
	import sys	
	formatter_path = r"..\..\MiniPys\Formatter"
	sys.path.insert(0, formatter_path)
	import minipy_formatter as MF
	plt.style.use('default')
	MF.Format().rcUpdate()


	tstep = 1
	N = 10000
	k = 10
	t = np.linspace(0, (N - 1) * tstep, N)
	y = np.sin(k * (2 * np.pi / N) * t)

	fstep = 1 / tstep
	f = np.linspace(0, (N - 1) * fstep, N)
	fft = np.fft.fft(y)

	fig, axs = plt.subplot_mosaic([['time', 'freq']])

	axs['time'].plot(t, y)
	axs['freq'].plot(f, np.abs(fft))
	plt.show()


if __name__ == '__main__':
	main()