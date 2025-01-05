import cv2 as cv
import os
from typing import Tuple
from numpy import ndarray
from circle import *

def getVideoWriter(video_dir: str, video_file: str) -> cv.VideoWriter:
	# Output parameters
	dimensions = int(videocap.get(3)), int(videocap.get(4))
	fps = videocap.get(5)
	encoder = cv.VideoWriter_fourcc(*'mp4v')

	# VideoWriter
	video_name, extension = video_file.split('.')
	video_name += '_tracked.' + extension
	output = cv.VideoWriter(os.path.join(video_dir, video_name),
		encoder, fps, dimensions)
	return output

def put_text(img: ndarray, text: str, position: Tuple[int, int]):
	output = cv.putText(img, f"{t:.2f}s", text_pos, cv.FONT_HERSHEY_TRIPLEX,
		1, (255, 255, 255), 1, cv.LINE_AA)
	return output

def get_circle(img: ndarray, show: bool = False, return_all: bool = False):
	'''Returns circle center and adds circle on image.'''
	radius = 27
	padding = 1
	threshold = 65
	kernel = build_kernel(radius, padding)
	detected = detect_circle(img, radius, threshold, kernel, show=show,
							 return_all=return_all)
	return detected

video_dir = './Images/06-12-24/2um_microparticles'
video_file = '2um_laser-browniano_3.mp4'

# Loading video
videocap = cv.VideoCapture(os.path.join(video_dir, video_file))
success, frame = videocap.read()
if not success:
	raise FileNotFoundError(f"'{video_path}' doesn't exist!")

time_step = 1 / videocap.get(5)
output = getVideoWriter(video_dir, video_file)
frame = get_circle(frame, True)[-1]
output.write(frame)

t = 0
i = 0
text_pos = (30, 30)
while success:
	success, frame = videocap.read()
	if not success:
		break
	frame = put_text(frame, f"{t:.2f}s", text_pos)
	frame = get_circle(frame)[1]
	i += 1
	t += time_step

# Release
output.release()
videocap.release()
