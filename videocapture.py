import cv2 as cv
import os
from typing import Tuple
from numpy import ndarray
import json
from circle import *

def get_VideoWriter(video_dir: str, video_file: str,
	videocap: cv.VideoCapture) -> cv.VideoWriter:
	# Output parameters
	dimensions = int(videocap.get(3)), int(videocap.get(4))
	fps = videocap.get(5)
	encoder = cv.VideoWriter_fourcc(*'mp4v')

	# VideoWriter
	video_name, extension = video_file.split('.')
	video_name += '_tracked.mp4'
	output = cv.VideoWriter(os.path.join(video_dir, video_name),
		encoder, fps, dimensions)
	return output

def write_text(img: ndarray, text: str, position: Tuple[int, int]):
	img = cv.putText(img, text, position, cv.FONT_HERSHEY_TRIPLEX,
		0.2, (255, 255, 255), 1, cv.LINE_AA)
	return img

def get_tracking_parameters(img_filename, json_path):
	with open(json_path, 'r') as f:
		data = json.load(f)
	data = data['main'][img_filename]
	global video_dir
	video_dir = data['dir']
	global radius, padding
	radius, padding = data['radius'], data['padding']
	global n_circles, threshold
	n_circles, threshold = data['n_circles'], data['threshold']
	global img_center, crop_D
	img_center, crop_D = data['img_center'], data['crop_D']
	global kernel
	kernel = build_kernel(radius, padding)

def get_circle(img: ndarray, last_centers = None, show: bool = False):
	'''Returns circle center and adds circle on image.'''
	detected = detect_circle(img, radius, radius, n_circles, threshold, kernel,
		show=show, img_center=img_center, crop_D=crop_D, last_centers=last_centers)
	return detected

def progress(i, N):
	percent = round(100 * i / N, 2)
	print(f"Processed {i} / {N} frames ({percent}%)")

def main():
	video_file = 'Key Facts.mp4'
	json_path = './Images/tracking.json'
	get_tracking_parameters(video_file, json_path)

	# Loading video
	videocap = cv.VideoCapture(os.path.join(video_dir, video_file))
	success, frame = videocap.read()
	if not success:
		raise FileNotFoundError(f"'{video_path}' doesn't exist!")

	# Text params
	total_frames = int(videocap.get(7))
	text_pos, t, i = (10, 20), 0, 0

	# Getting videowriter and first frame
	time_step = 1 / videocap.get(5)
	output = get_VideoWriter(video_dir, video_file, videocap)
	frame = write_text(frame, f"{i + 1} - {t:.2f}s", text_pos)
	last_centers, frame = get_circle(frame, None, True)[0:2]
	output.write(frame)
	while success:
		progress(i + 1, total_frames)
		success, frame = videocap.read()
		if not success:
			break
		frame = write_text(frame, f"{i + 1} - {t:.2f}s", text_pos)
		last_centers, frame = get_circle(frame, last_centers)[0:2]
		output.write(frame)
		i += 1
		t += time_step

	# Release
	output.release()
	videocap.release()

if __name__ == '__main__':
	main()
