import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_color(image_path, k=3):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # uklanjanje bijele pozadine
    mask = ~((s < 30) & (v > 200))
    pixels = img[mask]

    if len(pixels) == 0:
        return {"rgb": [0, 0, 0], "hex": "#000000"}

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    dominant = kmeans.cluster_centers_[np.argmax(
        np.bincount(kmeans.labels_)
    )]

    rgb = dominant.astype(int)
    hex_color = '#%02x%02x%02x' % tuple(rgb)

    return {
        "rgb": rgb.tolist(),
        "hex": hex_color
    }
