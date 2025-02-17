import cv2
import numpy as np
from src.feature_extraction.gabor_utils import calculate_lambda
import matplotlib.pyplot as plt
import logging
from skimage import feature, io, color

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_color_features(image, num_blocks):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv_image.shape
    block_height, block_width = h // num_blocks[0], w // num_blocks[1]
    features = []

    for row in range(0, h, block_height):
        for col in range(0, w, block_width):
            block = hsv_image[row:row+block_height, col:col+block_width]
            for channel in range(3):
                pixels = block[:, :, channel]
                mean = np.mean(pixels)
                std_dev = np.std(pixels)
                skewness = np.mean((pixels - mean) ** 3) / (std_dev ** 3 + 1e-8)
                features.append([mean, std_dev, skewness])

    return np.array(features)

def extract_gabor_features(image, kernel_size=(64, 64), num_orientation = 8, num_scales = 5):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = []

    for scale in range(num_scales):
        for theta in range(num_orientation):
            lamda = calculate_lambda(scale)
            kernel = cv2.getGaborKernel(kernel_size, 2 * np.pi, theta, lamda, 0.4)
            image_filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            mean = np.mean(image_filtered)
            var = np.var(image_filtered)
            skewness = np.mean((image_filtered - mean) ** 3 / (np.std(image_filtered) ** 3 + 1e-8))
            features.append([mean, var, skewness])

    return np.array(features)


def extract_gist_feature(image, num_scales = 4, num_orientations = 8):
    gabor_kernels = []
    image = cv2.resize(image, (55, 60))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for scale in range(num_scales):
        for orientation in range(num_orientations):
            theta = np.pi * orientation / num_orientations
            sigma = 0.56 * 2**scale
            lamda = np.pi * 2**scale
            kernel = cv2.getGaborKernel((15, 15), sigma, theta, lamda, 0.5, psi=0, ktype=cv2.CV_32F)
            gabor_kernels.append(kernel)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    h = image - blurred
    dft = cv2.dft(np.float32(h), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    gist_features = []
    
    for kernel in gabor_kernels:
        filtered = cv2.filter2D(dft_shifted[:, :, 0], -1, kernel)
        blocks = blockshaped(filtered, filtered.shape[0]//5, filtered.shape[1]//5)
        gist_features.extend([np.mean(block) for block in blocks])

    return np.array(gist_features)

def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    if h % nrows != 0 or w % ncols != 0:
        logging.error("Dimensioni non divisibili: {}x{} con blocchi {}x{}".format(h, w, nrows, ncols))
        return None
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def extract_eoh_features(image, num_bins = 37):

    edge = cv2.Canny(image, 100, 200)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    hist, _ = np.histogram(angle, num_bins, range=(0, 360), weights=mag)
    return hist / np.sum(hist)


def extract_lbp_features(image, n_points = 8, radius = 1):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = n_points * (n_points - 1) + 3
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=False)
    return np.array(hist)




