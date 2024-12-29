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

def extract_lbp_features(image, n_points = 8, radius = 1, n_bins = 59):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = 59
    hist, bins = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=False)
    
    features_array = np.array(hist)

    return features_array




