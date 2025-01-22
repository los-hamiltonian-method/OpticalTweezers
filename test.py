import json
import matplotlib.pyplot as plt
import cv2 as cv
import os
from videocapture import get_tracking_parameters, get_circle
import numpy as np
from circle import argmax2d
from typing import Tuple
ndarray = np.ndarray
#%%

'''
Planning on using center detected on center_filter and then averaging over a
distance w around it on img_cropped.
'''

video_file = '62-00mA_1.avi'
json_path = './Images/20-01-25/tracking.json'
with open(json_path, 'r') as f:
    data = json.load(f)
get_tracking_parameters(video_file, json_path)
full_path = os.path.join(data['main'][video_file]['dir'], video_file)

videocap = cv.VideoCapture(full_path)
_, frame = videocap.read()
_, _, img_cropped, _, _, center_filter = get_circle(frame, None, False)
cropped_center = argmax2d(img_cropped)
filter_center = argmax2d(center_filter)


#%% New algorithm
# Parameters
center = list(reversed(filter_center)) # (x, y)
img = img_cropped

def refine_center(img: ndarray, center: Tuple[float, float],
                  w_distance: float) -> Tuple[float, float]:
    '''
    Improves approximation of particle center with brightness-weighted
    displacements. Center must be given in (x, y).
    '''
    # Normalizing image
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    dim_y, dim_x = img.shape
    weighted_x = 1
    weighted_y = 1
    sum_intensity = 0
    i = -1
    
    # w-range
    while abs(weighted_x) > 0.5 or abs(weighted_y) > 0.5:
        if i > 100:
            break
        for y, row in enumerate(img_cropped):
            for x, intensity in enumerate(row):
                dx = x - center[0]
                dy = y - center[1]
                if dx**2 + dy**2 > w_distance**2:
                    continue
                sum_intensity += intensity
                weighted_x += dx * intensity
                weighted_y += dy * intensity
        weighted_x /= sum_intensity
        weighted_y /= sum_intensity
        i += 1
        center = center[0] + weighted_x, center[1] + weighted_x
    return center
    
    #plt.imshow(img_cropped)
    #x = np.linspace(center[0] - w_distance, center[0] + w_distance, 1000)
    #circle_plot = lambda x: np.sqrt(w_distance**2 - (x - center[0])**2)
    #plt.plot(x, circle_plot(x) + center[1], color='red')
    #plt.plot(x, -circle_plot(x) + center[1], color='red')
    #plt.scatter(*center, color='red', s=5)