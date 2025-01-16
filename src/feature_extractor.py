import cv2
import numpy as np
from utils.gabor_utils import calculate_lambda
import matplotlib.pyplot as plt
from skimage import feature, io, color

def extract_color_features(image, num_blocks):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, w, _ = hsv_image.shape

    block_height, block_width = h // num_blocks[0], w // num_blocks[1]

    features = []

    for row in range(0, h, block_height):
        for col in range(0, w, block_width):

            block = hsv_image[row:row+block_height, col:col+block_width]

            for channel in range(3):
                
                pixels = block[:,:,channel]

                mean = np.mean(pixels)

                std_dev = np.std(pixels)

                skewness = np.mean((pixels - mean) ** 3) / (std_dev ** 3 + 1e-8)

                features.append([mean, std_dev, skewness])

    features_array = np.array(features)

    return features_array

def extract_gabor_features(image, kernel_size=(64, 64), num_orientation = 8, num_scales = 5):

    k_max = np.pi / 2
    f = np.sqrt(2)
    sigma = 2 * np.pi
    gamma = 0.4
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = []

    for scale in range(num_scales):
        for theta in range(num_orientation):
            lamda = calculate_lambda(scale)
            kernel = cv2.getGaborKernel(kernel_size, sigma, theta, lamda, gamma)
            image_filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            mean = np.mean(image_filtered)
            var = np.var(image_filtered)
            skewness = np.mean((image_filtered - mean) ** 3 / (np.std(image_filtered) ** 3 + 1e-8))
            features.append([mean, var, skewness])

    features_array = np.array(features)

    return features_array

def extract_gist_feature(image, num_scales = 4, num_orientations = 8):
    gabor_kernels = []

    image = cv2.resize(image, (55, 60))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for scale in range(num_scales):
        for orientation in range(num_orientations):
            theta = np.pi * orientation / num_orientations
            sigma = 0.56 * 2**scale
            lamda = np.pi * 2**scale
            gamma = 0.5
            kernel = cv2.getGaborKernel((15, 15), sigma, theta, lamda, gamma, psi=0, ktype=cv2.CV_32F)
            gabor_kernels.append(kernel)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    h = image - blurred

    dft = cv2.dft(np.float32(h), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    gist_features = []
    for kernel in gabor_kernels:
        filtered = cv2.filter2D(dft_shifted[:,:,0], -1, kernel)
        blocks = blockshaped(filtered, filtered.shape[0]//5, filtered.shape[1]//5)
        gist_features.extend([np.mean(block) for block in blocks])

    return np.array(gist_features)

def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def extract_eoh_features(image, num_bins = 37):

    t_lower = 100  # Lower Threshold 
    t_upper = 200

    edge = cv2.Canny(image, t_lower, t_upper) 

    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

    hist, bin_edges = np.histogram(angle, num_bins, range=(0, 360), weights=mag)
    hist_normalized = hist / np.sum(hist)
  
    features_array = np.array(hist_normalized)

    return features_array

def extract_lbp_features(image, n_points = 8, radius = 1):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = n_points*(n_points-1)+3
    hist, bins = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=False)
    
    features_array = np.array(hist)

    return features_array




