import os
import cv2
import numpy as np

def load_data(data_path, num_samples_to_process):
    data_dir_list = os.listdir(data_path)
    processed_images = 0
    images = []

    for dataset in data_dir_list:
        dataset_path = os.path.join(data_path, dataset)
        
        if os.path.isdir(dataset_path):
            emotion_folders = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
            
            for emotion in emotion_folders:
                emotion_path = os.path.join(dataset_path, emotion)
                
                if os.path.exists(emotion_path):
                    img_list = os.listdir(emotion_path)
                    
                    if len(img_list) > 0:
                        # Randomly select 1 image per emotion folder
                        img = np.random.choice(img_list)
                        img_path = os.path.join(emotion_path, img)
                        
                        input_img = cv2.imread(img_path)
                        
                        if input_img is not None and input_img.shape[:2] != (0, 0):
                            images.append(input_img)
                            processed_images += 1
                            
                            if processed_images >= num_samples_to_process:
                                return images
    return images
