import numpy as np

def calculate_lambda(scale, k_max=np.pi / 2, f=np.sqrt(2)):
    return 2 * np.pi / (k_max / (f ** scale))