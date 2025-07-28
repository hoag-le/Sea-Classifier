import os
import numpy as np
from PIL import Image

def load_images(data_dir, img_size=(224, 224)):
    images = []
    labels = []
    class_names = sorted([name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(class_to_idx[class_name])
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return np.array(images), np.array(labels), class_names
