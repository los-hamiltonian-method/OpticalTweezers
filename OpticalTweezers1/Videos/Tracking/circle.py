import cv2 as cv
import os
import numpy as np

videopath = "../06-12-24/2um_microparticles"
filename = "2um_laser-browniano_3"
fullpath = os.path.join(videopath, filename + ".mp4")
videocap = cv.VideoCapture(fullpath)

frame_n = 1
# Particle is around 57 pixels.
# First threshold
param1 = 40
# Second threshold, highest is more strict in circle detection
param2 = 20
# Min and max detectable circle radius (px)
minRadius, maxRadius = 15, 60
# Minimum circle separation
minDist = 200

dimensions = np.uint16((videocap.get(3), videocap.get(4)))
original_fps = videocap.get(cv.CAP_PROP_FPS)
time_step = 1 / original_fps
# Output video
outputvid = cv.VideoWriter(f"{filename}_tracked.mp4",
	cv.VideoWriter_fourcc(*'mp4v'), original_fps, dimensions)

if videocap.isOpened():
	success, frame = videocap.read()
else:
	success = False

time = 0
text_pos = (50, 50)
while success:
	# Reading video
	success, frame = videocap.read()
	if not success:
		break
	
	# Circle detection
	# Frame must be converted to GRAY in order for Hough to work.
	frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	# image / algorithm / ratio of resolution of OG image to accumulator /
	# minDist between circles / first threshold detection param / second
	# threshold detector param / minRadius of circle / maxRadius of circle
	detected_circles = cv.HoughCircles(frame, cv.HOUGH_GRADIENT, 1, minDist,
		param1=param1, param2=param2, minRadius=minRadius,  maxRadius=maxRadius)

	# Around only rounds but keeps entries as floats.
	detected_circles = np.uint16(np.around(detected_circles))

	# Drawing circles
	# Changing back to color
	frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
	# Order for detected circles is x-pos, y-pos, radius.
	for circle in detected_circles[0, :]:
		a, b, r = circle
		# Circle
		cv.circle(frame, (a, b), r, (0, 255, 255), 2)
		# Circle center
		cv.circle(frame, (a, b), 3, (0, 0, 255), -1)


	frame = cv.putText(frame, f"t = {time:.2f}s", text_pos, cv.FONT_HERSHEY_TRIPLEX,
		1, (255, 255, 255), 1, cv.LINE_AA)

	cv.waitKey(0)

	outputvid.write(frame)

	time += time_step
	#cv.imshow('Writing Video', frame)

	#if cv.waitKey(1) == 27: 
	#	break

cv.destroyAllWindows()
outputvid.release()
videocap.release()