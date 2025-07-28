import sys
sys.path.append('.')

from tensorflow import keras
import numpy as np
from PIL import Image
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'sea_classifier.h5')
CLASS_FILE = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_names.txt')

def load_class_names():
    with open(CLASS_FILE, 'r') as f:
        return [line.strip() for line in f]

def predict_image(image_path):
    model = keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    args = parser.parse_args()

    pred, conf = predict_image(args.img_path)
    print(f"Predicted: {pred} (Confidence: {conf:.2f})")
