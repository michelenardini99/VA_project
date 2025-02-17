import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.stats import skew
from skimage.feature import local_binary_pattern

def calculate_lambda(scale, k_max=np.pi / 2, f=np.sqrt(2)):
    return 2 * np.pi / (k_max / (f ** scale))


