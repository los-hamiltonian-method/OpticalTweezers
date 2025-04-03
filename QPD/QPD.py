import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def mainloop(dsname, reference_name, current, reference, pds_avg):
	# I/O
	reference_data = pd.read_csv(dsname, skiprows=27, names=['time(s)', 'voltage(V)', 'nan'])
	data = pd.read_csv(dsname, skiprows=27, names=['time(s)', 'voltage(V)', 'nan'])
	print(dsname)
	print(data.describe(), '\n')
	t = data['time(s)'][:len(data) // 2]
	V = data['voltage(V)'][:len(data) // 2]

	# V = 3 * np.sin(60 * 2 * np.pi * t)

	# FFT parameters
	tstep = round(t[1] - t[0], 5)
	N = len(t)
	Fs = 1 / tstep # sample frequency
	fstep = Fs / N
	f = np.linspace(0, (N - 1) * fstep, N)
	fft = np.fft.fft(V)
	fft_abs = np.abs(fft) / N

	# One sided, skipping 2DC term
	fft_amp = 2 * fft_abs / N
	fft_pds = 2 * fft_abs**2 / N**2
	fft_amp, fft_pds = fft_amp[1:N // 2], fft_pds[1:N // 2]
	f = f[1:N // 2]


	# Plots
	mosaic = [['time', 'freq']]
	fig, axs = plt.subplot_mosaic(mosaic, figsize=(15, 10))
	suptitle = f"Current: {current}mA\n File: {dsname} (cut)"
	if reference: suptitle = 'Reference - ' + suptitle
	fig.suptitle(suptitle)
	
	if reference:
		pds_avg = fft_pds
		global reference_fft_amp, reference_fft_pds
		reference_fft_amp = fft_amp
		reference_fft_pds = fft_pds
	else: pds_avg += fft_pds

	# Time domain
	axs['time'].plot(t, V, lw=0.5, label='$x$-diff from QPD')
	axs['time'].set(title='Time domain', xlabel='Time [s]', ylabel='Voltage [V]')
	axs['time'].grid(True)

	# Frequency domain
	# axs['freq'].scatter(f, reference_fft_pds, s=3, label='Reference PDS of $x$-diff', alpha=0.5)
	axs['freq'].scatter(f, fft_pds, s=3, label='PDS of $x$-diff')
	# axs['freq'].scatter(f, reference_fft_pds, s=1, label='AMP of reference $x$-diff')
	axs['freq'].set(title='Frequency domain', xlabel='Frequency [Hz]', ylabel='Amplitude')
	axs['freq'].grid(True)
	axs['freq'].set(xscale='log', yscale='log')

	for ax in axs: axs[ax].legend()

	# Comment out if ¬Formatter
	plt.savefig(dsname.replace('.CSV', '_cut.png'), dpi=200)
	# Format.fullscreen_show()
	# plt.show()
	return pds_avg

def main():
	# If you don't have Formatter comment out the following lines.
	import sys	
	formatter_path = r"..\..\MiniPys\Formatter"
	sys.path.insert(0, formatter_path)
	import minipy_formatter as MF
	plt.style.use('default')
	global Format
	Format = MF.Format()
	Format.rcUpdate()

	dirpath = r'.\data\11-03-25\renamed'
	listdir = [ds for ds in os.listdir(dirpath) if not 'SKIP_' in ds]

	pds_avg = np.zeros(4999)
	reference_name = ''
	for ds in listdir:
		if '.png' in ds: continue
		reference = False
		dsname = os.path.join(dirpath, ds)
		current = dsname.split('_')[1].replace('.CSV', '').replace('mA', '') # mA
		if 'reference' in ds:
			reference_name = dsname
			reference = True
		pds_avg = mainloop(dsname, reference_name, current, reference, pds_avg)


if __name__ == '__main__':
	main()