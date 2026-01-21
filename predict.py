import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
import os

# ===== POSTAVKE =====
MODEL_PATH = "fashion_mobilenetv2_finetuned.keras"
IMG_SIZE = (224, 224)
IMAGE_PATH = "test_images/3171443_2.jpg"

# ===== UČITAJ MODEL =====
model = tf.keras.models.load_model(MODEL_PATH)

# ===== LOAD CLASS NAMES =====
# OVO MORA BITI ISTI REDOSLIJED KAO U TRENINGU
class_names = [
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
# ⬆️ prilagodi točno svojim folderima!

# ===== UČITAJ SLIKU =====
img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # normalizacija
img_array = np.expand_dims(img_array, axis=0)

# ===== PREDIKCIJA =====
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions)
confidence = predictions[0][predicted_index]

predicted_class = class_names[predicted_index]

# ===== ISPIS =====
print("Predikcija:")
print(f"Klasa: {predicted_class}")
print(f"Pouzdanost: {confidence:.2f}")
