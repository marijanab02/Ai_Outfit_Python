import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = "models/fashion_mobilenetv2_finetuned.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    'jackets',
    'long_sleeve_dress',
    'long_sleeve_top',
    'short_sleeve_dress',
    'short_sleeve_top',
    'shorts',
    'skirt',
    'trousers',
    'vest',
    'vest_dress'
]

# ⚡ učitaj model pri startupu
model = tf.keras.models.load_model(MODEL_PATH)

def classify_image(image_path: str):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    idx = np.argmax(preds)
    
    return {
        "category": CLASS_NAMES[idx],
        "confidence": float(preds[0][idx])
    }
