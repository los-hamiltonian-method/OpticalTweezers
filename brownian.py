import matplotlib.pyplot as plt
import numpy as np
import json
import sys
formatter_path = r"..\MiniPys\Formatter"
sys.path.insert(0, formatter_path)
import minipy_formatter as MF
plt.style.use('default')
MF.Format().rcUpdate()

video_file = '63-00mA_1.avi'
json_path = './Images/20-01-25/tracking.json'
with open(json_path, 'r') as f:
    data = json.load(f)['main'][video_file]
    
centers = np.array(data['centers'])[:,0,:]
num_centers = len(centers)
time_step = data['time_step']
times = np.array([i * time_step for i in range(0, num_centers)])
x_centers = centers[:, 0]
y_centers = centers[:, 1]

# Shifting
x_centers -= x_centers.mean()
y_centers -= y_centers.mean()

fig, axs = plt.subplot_mosaic([['x'], ['y']], dpi=200)
labels = ['x', 'y']
axs['x'].plot(times, x_centers, linewidth=0.5)
axs['y'].plot(times, y_centers, color='red', linewidth=0.5)

fig.suptitle(f"Brownian motion for {video_file}")
for c, label in zip((x_centers, y_centers), axs):
    axs[label].set(xlabel=f'Time (t), step={round(time_step, 4)}',
                   ylabel=f'${label}$-displacement')
    axs[label].axhline(0, color='#888', linestyle='--', linewidth=1)
    axs[label].axhline(np.max(c), color='#888', linestyle='--', linewidth=1)
    axs[label].axhline(np.min(c), color='#888', linestyle='--', linewidth=1)

# TODO: Save figs.