import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ================= PUTANJE =================
CSV_PATH = "images_compressed/images.csv"
IMAGES_DIR = "images_compressed"
OUTPUT_DIR = "dataset2"

TRAIN_RATIO = 0.8
MIN_SAMPLES_PER_CLASS = 10

IMAGE_EXT = ".jpg"   # <-- ako su png, promijeni u ".png"

# ================= UČITAJ CSV =================
df = pd.read_csv(CSV_PATH)

df = df[['image', 'label', 'kids']]
df.dropna(inplace=True)

# ================= FILTER: MAKNUTI DJEČJU ODJEĆU =================
df = df[df['kids'] == False]
print(f"Nakon uklanjanja kids=True: {len(df)}")

# ================= DODAJ EKSTENZIJU =================
df['filename'] = df['image'].astype(str) + IMAGE_EXT

# ================= FILTER: POSTOJEĆE SLIKE =================
def image_exists(filename):
    return os.path.isfile(os.path.join(IMAGES_DIR, filename))

df = df[df['filename'].apply(image_exists)]
print(f"Nakon provjere slika: {len(df)}")

# ================= UKLONI RIJETKE KLASE =================
class_counts = df['label'].value_counts()
valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
df = df[df['label'].isin(valid_classes)]

print(f"Broj klasa nakon filtriranja: {df['label'].nunique()}")
print(f"Ukupno slika nakon filtriranja: {len(df)}")

# ================= SPLIT =================
train_df, val_df = train_test_split(
    df,
    test_size=1 - TRAIN_RATIO,
    stratify=df['label'],
    random_state=42
)

# ================= ČIŠĆENJE NAZIVA =================
def clean_label(label):
    return (
        label.lower()
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("&", "and")
    )

# ================= KOPIRANJE =================
def copy_images(dataframe, split):
    for _, row in dataframe.iterrows():
        label = clean_label(row['label'])
        filename = row['filename']

        src = os.path.join(IMAGES_DIR, filename)
        dst_dir = os.path.join(OUTPUT_DIR, split, label)
        dst = os.path.join(dst_dir, filename)

        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst)

    print(f"{split.upper()} set gotov ✔")

copy_images(train_df, "train")
copy_images(val_df, "val")

print("🎉 Dataset uspješno pripremljen!")
