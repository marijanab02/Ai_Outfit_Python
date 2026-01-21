import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def extract_dominant_color(image_path, k=3):
    # 1️⃣ učitaj sliku
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2️⃣ resize (brže i stabilnije)
    image = cv2.resize(image, (300, 300))

    # 3️⃣ RGB → HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # 4️⃣ maska za uklanjanje bijele pozadine
    # bijela = low saturation + high value
    mask = ~((s < 30) & (v > 200))

    # 5️⃣ izvuci samo piksele odjeće
    pixels = image[mask]

    if len(pixels) == 0:
        raise ValueError("Nema pronađenih piksela nakon maskiranja!")

    pixels = pixels.reshape((-1, 3))

    # 6️⃣ KMeans
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)

    # 7️⃣ dominantni klaster
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]

    return dominant_color.astype(int)


def show_color(color):
    patch = np.zeros((100, 100, 3), dtype="uint8")
    patch[:] = color

    plt.imshow(patch)
    plt.axis("off")
    plt.title(f"Dominantna boja: {color}")
    plt.show()


# ▶️ PRIMJER POZIVA
image_path = "test_images/1feacab8-31e0-4861-b424-1ec78acada5a.jpg"  # <- promijeni putanju

dominant_color = extract_dominant_color(image_path, k=3)
print("Dominantna boja (RGB):", dominant_color)

show_color(dominant_color)
