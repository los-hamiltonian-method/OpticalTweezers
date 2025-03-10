'''Circle detection algorithm.'''
#%% Imports
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from copy import deepcopy

ndarray = np.ndarray
optuple = Optional[Tuple[int, int]]


#%% Main functions
def detect_edges(img: ndarray, threshold: int, center: optuple = None,
    crop_D: optuple = None) -> Tuple[ndarray, ndarray]:
    '''
    Detects edges on image array by converting image to binary map and then
    applying and edge detection kernel.
    If shown, use keys to destroy windows.
    Parameters
    ----------
    img: ndarray
        Image array.
    center: Optional[Tuple[int, int]]
        (y, x) position from which to crop image.
    crop_D: Optional[Tuple[int, int]]
        (y, x) distance from center from which to crop image.
        If None, let's image as is.
    threshold: int
        Minimum value for which gray scale image pixels are converted
        to white on binary map.
    show: bool
        Show binary map and identified edges.

    Returns
    -------
    img_bw: ndarray
        ndarray of black and white image. Same dimensions as
        cropped image.
    edges: ndarray
        ndarray containing edges of cropped image. Same dimensions as
        cropped image.
    '''
    if center == None:
        # For an odd image, center is a 1x1 square.
        # For an even image, center is a 2x2 square.
        # For a combination, it is a 2x1 or 1x2 rectangle.
        # I consider the center to be the upper left corner
        # of this rectangle.
        shape = np.array(img.shape) - 1
        center = shape // 2

    if crop_D != None:
        crop_D = np.array(crop_D)
        img = img[center[0] - crop_D[0]:center[0] + crop_D[0] + 1,
                    center[1] - crop_D[1]:center[1] + crop_D[1] + 1]

    # Converting image to binary map with threshold.
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # TODO: Is there any gain in letting configure 255 upper bound?
    # I think yes.
    # TODO: Colored image support. Take differences in pixels.
    img_bw = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)[1]

    # TODO: I think I can do this with a convolution composition
    # and make a pass with a single array.
    y_kernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    x_kernel = np.transpose(y_kernel)
    y_deriv = cv.filter2D(src=img_bw, ddepth=-1, kernel=y_kernel)
    x_deriv = cv.filter2D(src=img_bw, ddepth=-1, kernel=x_kernel)
    inv_y_deriv = cv.filter2D(src=img_bw, ddepth=-1, kernel=-y_kernel)
    inv_x_deriv = cv.filter2D(src=img_bw, ddepth=-1, kernel=-x_kernel)
    edges = np.sqrt(x_deriv**2 + y_deriv**2
                    + inv_y_deriv**2 + inv_x_deriv**2).astype(float)
    edges = 255 * edges / np.max(edges)
    return img, img_bw, edges

def build_kernel(radius: int = 100, padding: int = 5) -> ndarray:
    '''Builds a single disk kernel array.'''
    min_R = radius - padding
    max_R = radius + padding

    # Substracting max_R as it is position of circles center on img array.
    Filter = lambda i, j: min_R < np.sqrt((i - max_R)**2
                         + (j - max_R)**2) < max_R
    kernel = [[Filter(i, j) for i in range(2 * max_R)]
              for j in range(2 * max_R)]
    kernel = np.array(kernel).astype(float)
    # Weighting kernel based on center circle.
    kernel = kernel / (2 * np.pi * radius)
    return kernel

def filter_center(edges: ndarray, kernel: ndarray,
    threshold: int = None) -> Tuple[ndarray, ndarray]:
    '''Extracts circle center based on passes from disk kernel.'''

    center = cv.filter2D(src=edges, ddepth=-1, kernel=kernel)
    center_bw = cv.threshold(center, threshold, 255,
                                cv.THRESH_BINARY)[1]
    return center, center_bw

def argmax2d(array: np.ndarray) -> Tuple[int, int]: #(x, y)
    '''Returns (y, x) positions of maximum value in array.'''
    arg = array.argmax()
    x_shape = array.shape[1]
    row = int(arg // x_shape)
    column = int(arg - row * x_shape)
    return column, row

# TODO: I'm not using threshold2 nor center_bw as of now.
# TODO: Change calls to Array-like[int, int]
# TODO: (y, x) for numpy, (x, y) for cv is a bit confusing. ぜんぶの[::-1]は
# これのせいです。
# TODO: Add return type.
# TODO: Perhaps add ability to change w_distance (as of now, w_d = radius)
lc_type = Optional[List[ndarray]]
def detect_circle(img: ndarray, radius: int, separation_D: int,
    n_circles: int, threshold: int, kernel=ndarray,
    threshold2: int = None, last_centers: lc_type = None,
    img_center: optuple = None, crop_D: optuple = None,
    refine=False, w_distance: float = None, show=False):
    '''Returns (x, y) coordinates of circle on img array.'''
    if not w_distance: w_distance = radius
    if bool(crop_D) != bool(img_center):
        raise ValueError("In order to crop the image both a crop distance "
                         "and an image center must be provided.")
    center_modifier = np.array([0, 0])
    if crop_D:
        center_modifier = np.array(img_center) - np.array(crop_D)
        # TODO: (y, x)!!!!!!!!!!!!!!!!

    img_cropped, img_bw, edges = detect_edges(img, center=img_center,
                                           crop_D=crop_D, threshold=threshold)
    center_filter = filter_center(edges, kernel)[0]
    if n_circles == 1:
        center = argmax2d(center_filter) # TODO: (x, y)!!!!!!!!!!!
        if refine: center = refine_center(img_cropped, center, w_distance)
        center = (center + center_modifier[::-1])
        centers = [center]
        img = draw_circle(img, center.astype(int), radius)
        img = draw_circle(img, center.astype(int), w_distance)
    else:
        centers, img = get_centers(img, center_filter, radius, separation_D,
                        n_circles, center_modifier, last_centers, refine)

    if show:
        show_circle(img, img_bw, edges, kernel, center_filter)
    return centers, img, img_cropped, img_bw, edges, center_filter

def distance_squared(array1, array2):
    '''Returnes square of separation distance between array1 and array2.'''
    x_2 = np.vectorize(lambda x: x**2)
    return sum(x_2(array1 - array2))

def refine_center(img: ndarray, center: Tuple[float, float],
                  w_distance: float) -> Tuple[float, float]:
    '''
    Improves approximation of particle center with brightness-weighted
    displacements. Center must be given in (x, y).
    '''
    # Normalizing image
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    weighted_x = 1
    weighted_y = 1
    sum_intensity = 0
    i = 0
    #plt.imshow(img)
    while abs(weighted_x) > 0.1 or abs(weighted_y) > 0.1:
        #plt.scatter(*center, s=2, color='blue')
        recursion_limit = 100
        if i > recursion_limit:
            print("Warning: In refining center the recursion limit "
                  f"({recursion_limit}) was exceeded.")
            break
        for y, row in enumerate(img):
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
    if i > 0: print(f"Refined center {i + 1} times.")
    
    #x = np.linspace(center[0] - w_distance, center[0] + w_distance, 1000)
    #circle_plot = lambda x: np.sqrt(w_distance**2 - (x - center[0])**2)
    #plt.plot(x, circle_plot(x) + center[1], color='red')
    #plt.plot(x, -circle_plot(x) + center[1], color='red')
    #plt.scatter(*center, color='red', s=5)
    return center

# TODO: Perhaps change the order of parameters.
# TODO: This is rather slow. Why? I think it's always slow on
# micro3 when I analyze the whole image; kernel is too small compared
# to image dimensions.
def get_centers(img: ndarray, center_filter: ndarray, radius: int,
    separation_D: int, n_circles: int, center_modifier: ndarray,
    last_centers: lc_type = None, refine=False):
    if refine: refined_centers = []
    centers = []
    center_filter_holed = deepcopy(center_filter)
    for _ in range(n_circles):
        center = argmax2d(center_filter_holed)[::-1]
        center_filter_holed = cv.circle(center_filter_holed, center,
            radius=separation_D, color=(0, 0, 0), thickness=-1)
        if refine:
            refined_centers.append(refine_center(center_filter_holed, center,
                                                 separation_D))
        center += center_modifier[::-1]
        # TODO: Change draw_circle so that it draws with matplotlib, allowing
        # for subpixel resolution.
        img = draw_circle(img, center, radius)
        centers.append(center)

    if not last_centers:
        key = lambda c: c[0]**2 + c[1]**2
        centers.sort(key=key)
    # TODO: Update heuristic so that it works with refined centers.
    else:
        # Heuristic for determining circle position continuity
        # when analyzing videos.
        ordered_centers = n_circles * [None]
        last_centers = list(enumerate(last_centers))
        for center in centers:
            key = lambda lcenter: distance_squared(center, lcenter[1])
            last_centers.sort(key=key)
            ordered_centers[last_centers.pop(0)[0]] = center
        centers = ordered_centers
    for i, center in enumerate(centers):
        position = (np.array(center)).astype(int)
        img = cv.putText(img, str(i), position, cv.FONT_HERSHEY_TRIPLEX,
                         0.3, (255, 255, 255), 1, cv.LINE_AA)
    return refined_centers, img

def draw_circle(img: ndarray, center: Tuple[int, int], radius: int):
    img = cv.circle(img, center, radius=0, color=(0, 0, 255),
                    thickness=3)
    img = cv.circle(img, center, radius=radius, color=(255, 255, 255),
                    thickness=1)
    return img

def show_circle(img, img_bw, edges, kernel, center_filter):
    plt.style.use('dark_background')
    # Adding circle to image
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Mosaic
    mosaic = [['img_bw', 'edges', 'img', 'img'],
              ['kernel', 'center_filter', 'img', 'img']]
    keys = ['img_bw', 'edges', 'kernel', 'center_filter', 'img']
    titles = ['Binary map', 'Edges', 'Kernel', 'Filtered center',
              'Original Image']
    imgs = [img_bw, edges, kernel, center_filter, img]

    # Plots
    fig, ax = plt.subplot_mosaic(mosaic, dpi=150, figsize=(8, 4.5))
    fig.suptitle('Circle Identification')
    fig.patch.set_facecolor('#787878')
    for key, im, title in zip(keys, imgs, titles):
        ax[key].imshow(im)
        ax[key].set(title=title)

    # Show
    plt.subplots_adjust(hspace=0, wspace=0.35)
    figManager = plt.get_current_fig_manager()
    figManager.full_screen_toggle()
    plt.show()


#%% main()
def main():
    import os
    import sys
    sys.path.insert(0, '../MiniPys/Formatter')
    import minipy_formatter as MF
    MF.Format().rcUpdate()
    # Images
    # img_name = 'micro.png'
    # img_name = 'micro2.bmp'
    # img_name = 'eye.webp'
    # radius
    # micro: 42, 1, 81
    # micro 2: 67, 1, 78
    # iris: 150, 1, 140
    # pupil: 61, 1, 40

    img_dir = './Circle detection/images/'
    img_name = 'micro2.bmp'
    img = cv.imread(os.path.join(img_dir, img_name))
    
    # Kernel params
    radius = 67
    padding = 1
    kernel = build_kernel(radius=radius, padding=padding)

    # Params
    n_circles = 1
    threshold = 78
    img_center = None
    crop_D = None

    detect_circle(img, radius=radius, n_circles=n_circles, separation_D=radius,
                  kernel=kernel, threshold=threshold,
                  img_center=img_center, crop_D=crop_D, show=True)


#%% 呪い
if __name__ == '__main__':
    main()
    