# FBI-PP
Fractal-Based Image Enhancement and Colorization Pipeline for Improved Visual Quality and Quantitative Evaluation

This Python-based image processing pipeline enhances grayscale images using advanced techniques like fractal patterns, edge detection, and noise reduction. Designed for medical imaging, satellite imagery, and facial recognition applications, it improves image quality for better analysis and decision-making. The code is modular, divided into three components dataloader.py, image_enhance.py, and main.py, ensuring reusability and scalability.

The dataloader.py script loads images from a dataset directory, randomly selecting a specified number for processing. The image_enhance.py script performs the core enhancement, using Gaussian blur, Sobel edge detection, CLAHE (Contrast Limited Adaptive Histogram Equalization), and fractal patterns generated via Perlin noise. These techniques amplify details, preserve edges, and improve overall image quality. For qualitative results, a simple colorization method is also applied to the enhanced images using OpenCV's colormap functionality.

The main.py script orchestrates the process, calculating quantitative metrics like PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), MSE (Mean Squared Error), and EPI (Edge Preservation Index) to evaluate enhancement quality. Results, including original, enhanced, and colorized images, are visualized using Matplotlib, clearly comparing improvements.

This pipeline is a powerful tool for researchers and developers. It combines fractal-based enhancement, edge preservation, and quantitative evaluation. Its modular design and real-world applicability make it ideal for computer vision tasks where image quality is critical.

We are using the OULU-CASIA NI weak dataset for a robust evaluation and weak represents that the images are taken in low-light conditions. 

Qualitative and Quantitative results are as follows:

![image](https://github.com/user-attachments/assets/43efea00-de15-4022-87fc-afd9c1a95e32)
![image](https://github.com/user-attachments/assets/e7e5c61d-022d-441b-abd4-3733761ff620)
![image](https://github.com/user-attachments/assets/fb33071a-0956-439a-9f5b-61b13deb8663)
