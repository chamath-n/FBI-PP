import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from dataloader import load_data
from image_enhance import fractal_image_enhancement, infer_color

data_path = 'Oulu_CASIA_NIR_VIS/NI/Weak_1'  # Path to your dataset
img_rows, img_cols = 256, 256  
num_samples_to_process = 4  

def calculate_psnr(original, enhanced):
    return psnr(original, enhanced, data_range=enhanced.max() - enhanced.min())

def calculate_ssim(original, enhanced):
    return ssim(original, enhanced, data_range=enhanced.max() - enhanced.min())

def calculate_mse(original, enhanced):
    return np.mean((original - enhanced) ** 2)

def calculate_epi(original, enhanced):
    original_edges = cv2.Canny(original, 100, 200)
    enhanced_edges = cv2.Canny(enhanced, 100, 200)
    return np.sum(original_edges * enhanced_edges) / np.sum(original_edges)

def display_results(original, enhanced, colorized):
    # Resize the enhanced image to match the original dimensions
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enhanced_resized = cv2.resize(enhanced, (original_gray.shape[1], original_gray.shape[0]))
    colorized_resized = cv2.resize(colorized, (original_gray.shape[1], original_gray.shape[0]))

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(enhanced_resized, cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(colorized_resized, cv2.COLOR_BGR2RGB))
    plt.title('Colorized Image')
    plt.axis('off')
    
    plt.show()

    psnr_value = calculate_psnr(original_gray, enhanced_resized)
    ssim_value = calculate_ssim(original_gray, enhanced_resized)
    mse_value = calculate_mse(original_gray, enhanced_resized)
    epi_value = calculate_epi(original_gray, enhanced_resized)

    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"MSE: {mse_value:.2f}")
    print(f"EPI: {epi_value:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    images = load_data(data_path, num_samples_to_process)
    
    for input_img in images:
        enhanced_img = fractal_image_enhancement(input_img)
        colorized_img = infer_color(enhanced_img)
        display_results(input_img, enhanced_img, colorized_img)
