import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ================= PUTANJE =================
CSV_PATH = "metadata.csv"
IMAGES_DIR = "images"
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.8
MIN_SAMPLES_PER_CLASS = 10   # možeš povećati kasnije

# ================= UČITAJ CSV =================
df = pd.read_csv(CSV_PATH)

# koristimo garment kao labelu
df = df[['filename', 'garment']]
df.dropna(inplace=True)

# ================= FILTER: POSTOJEĆE SLIKE =================
def image_exists(filename):
    return os.path.exists(os.path.join(IMAGES_DIR, filename))

df = df[df['filename'].apply(image_exists)]

print(f"Ukupno validnih slika: {len(df)}")

# ================= UKLONI RIJETKE KLASE =================
class_counts = df['garment'].value_counts()
valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index

df = df[df['garment'].isin(valid_classes)]

print(f"Broj klasa nakon filtriranja: {df['garment'].nunique()}")
print(f"Ukupno slika nakon filtriranja: {len(df)}")

# ================= SPLIT TRAIN / VAL =================
train_df, val_df = train_test_split(
    df,
    test_size=1 - TRAIN_RATIO,
    stratify=df['garment'],
    random_state=42
)

# ================= ČIŠĆENJE NAZIVA FOLDERA =================
def clean_label(label):
    return (
        label.lower()
             .replace(" ", "_")
             .replace("/", "_")
             .replace("&", "and")
    )

# ================= KOPIRANJE SLIKA =================
def copy_images(dataframe, split_name):
    for _, row in dataframe.iterrows():
        label = clean_label(row['garment'])
        image_name = row['filename']

        src_path = os.path.join(IMAGES_DIR, image_name)
        dst_dir = os.path.join(OUTPUT_DIR, split_name, label)
        dst_path = os.path.join(dst_dir, image_name)

        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(src_path, dst_path)

    print(f"{split_name.upper()} set gotov ✔")

# ================= POKRENI =================
copy_images(train_df, "train")
copy_images(val_df, "val")

print("Dataset je uspješno pripremljen 🎉")
