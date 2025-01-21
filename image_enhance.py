import cv2
import numpy as np
from noise import pnoise2  

def fractal_image_enhancement(image):
    
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    low_freq = cv2.GaussianBlur(gray_image, (15, 15), 0)  

    high_freq = cv2.subtract(gray_image, low_freq)

    high_freq_enhanced = cv2.addWeighted(high_freq, 5.0, np.zeros_like(high_freq), 0, 0)  

    enhanced_image = cv2.addWeighted(gray_image, 0.8, high_freq_enhanced, 0.2, 0)

    sobel_x = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=3)  
    sobel_y = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=3)  
    sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)  
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  

    enhanced_image = cv2.addWeighted(enhanced_image, 0.8, sobel_edges, 0.3, 0)

    denoised_image = cv2.fastNlMeansDenoising(enhanced_image, h=2, templateWindowSize=3, searchWindowSize=2)

    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(16, 16))
    denoised_image = clahe.apply(denoised_image)

    rows, cols = denoised_image.shape
    scales = [0.02, 0.01, 0.0058]
    fractal_patterns = [generate_fractal_pattern((rows, cols), scale=s) for s in scales]

    fractal_patterns = [pattern.astype(np.float32) for pattern in fractal_patterns]

    enhanced_image = denoised_image.astype(np.float32)
    weights = [0.01, 0.01, 0.01] 

    for pattern, weight in zip(fractal_patterns, weights):
        enhanced_image = cv2.addWeighted(enhanced_image, 1, pattern, weight, 0)

    enhanced_image = cv2.bilateralFilter(enhanced_image, d=4, sigmaColor=100, sigmaSpace=50)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)

    enhanced_image = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return enhanced_image

def generate_fractal_pattern(shape, scale):
    pattern = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pattern[i, j] = pnoise2(i * scale, j * scale, octaves=6)
    return cv2.normalize(pattern, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def infer_color(enhanced_image):

    color_image = cv2.applyColorMap(enhanced_image, cv2.COLORMAP_VIRIDIS)
    return color_image
