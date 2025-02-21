import cv2 as cv
import os
from typing import Tuple
from numpy import ndarray
import json
from circle import *
from pprint import pprint

def get_VideoWriter(video_dir: str, video_file: str,
    videocap: cv.VideoCapture) -> cv.VideoWriter:
    # Output parameters
    dimensions = int(videocap.get(3)), int(videocap.get(4))
    fps = videocap.get(5)
    encoder = cv.VideoWriter_fourcc(*'MP42')

    # VideoWriter
    video_name, extension = video_file.split('.')
    video_name += '_tracked.' + extension
    output = cv.VideoWriter(os.path.join(video_dir, video_name),
        encoder, fps, dimensions)
    return output

def save_img(video_dir: str, video_file: str, img: ndarray) -> None:
    video_name, _ = video_file.split('.')
    video_name += '_frame.png'
    video_path = os.path.join(video_dir, video_name)
    cv.imwrite(video_path, img)

def write_text(img: ndarray, text: str, position: Tuple[int, int]):
    img = cv.putText(img, text, position, cv.FONT_HERSHEY_TRIPLEX,
        0.5, (255, 255, 255), 1, cv.LINE_AA)
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
    global start_frame, end_frame
    start_frame = data['start_frame']
    end_frame = data['end_frame']

def get_circle(img: ndarray, last_centers = None, show: bool = False):
    '''Returns circle center and adds circle on image.'''
    detected = detect_circle(img, radius, radius, n_circles, threshold, kernel,
        show=show, img_center=img_center, crop_D=crop_D,
        last_centers=last_centers, refine=True)
    return detected

def progress(i, N):
    percent = round(100 * i / N, 2)
    print(f"Processed {i} / {N} frames ({percent}%)")

def main():
    import sys
    formatter_path = r"..\MiniPys\Formatter"
    sys.path.insert(0, formatter_path)
    import minipy_formatter as MF
    MF.Format().rcUpdate()

    #files = ["62-00mA_1.avi", "63-00mA_2.avi", "63-50mA_1.avi", "63-50mA_2.avi", 
    #         "63-75mA_1.avi", "64-00mA_1.avi", "64-00mA_2.avi", "64-00mA_3.avi",
    #         "64-50mA_1.avi", "65-00mA_1.avi", "65-50mA_1.avi", "66-00mA_1.avi"]
    #for video_file in files:
    video_file = '66-00mA_1.avi'
    json_path = './Images/20-01-25/tracking.json'
    get_tracking_parameters(video_file, json_path)

    # Loading video
    text_pos, t, i = (10, 20), 0, 0
    videocap = cv.VideoCapture(os.path.join(video_dir, video_file))

    while i <= start_frame:
        success, frame = videocap.read()
        if not success:
            raise FileNotFoundError(f"'{video_path}' doesn't exist!")
        i += 1

    # Text params
    total_frames = int(videocap.get(7))
    global end_frame
    end_frame = min(total_frames, end_frame)

    # Getting videowriter and first frame
    # TODO: If null start or end, code the expected behaviour.
    # TODO: Progress bar might still be broken.

    time_step = 1 / videocap.get(5)
    output = get_VideoWriter(video_dir, video_file, videocap)
    last_centers, frame = get_circle(frame, None)[0:2]
    frame = write_text(frame, f"{i - 1} - {t:.2f}s", text_pos)
    #save_img(video_dir, video_file, frame)
    output.write(frame)
    
    
    centers = []
    while True:
        centers.append(np.array(last_centers).tolist())
        progress(i - start_frame, (end_frame + 1) - start_frame)
        success, frame = videocap.read()
        if not success or i > end_frame:
            break
        frame = write_text(frame, f"{i} - {t:.2f}s", text_pos)
        if i % 300 == 0.1: show = True
        else: show = False
        last_centers, frame = get_circle(frame, last_centers, show)[0:2]
        output.write(frame)
        i += 1
        t += time_step
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    data['main'][video_file]['time_step'] = time_step
    data['main'][video_file]['centers'] = centers # []
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    # Release
    output.release()
    videocap.release()
    
    

if __name__ == '__main__':
    main()
