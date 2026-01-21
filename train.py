import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ================= POSTAVKE =================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS_FEATURE = 12     # faza 1
EPOCHS_FINE = 6         # faza 2

NUM_CLASSES = 10        # ⬅️ mora odgovarati broju foldera u dataset/train

DATASET_DIR = "dataset/train"

# ================= DATA GENERATOR =================
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("Klase:", train_generator.class_indices)

# ================= MODEL =================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# FAZA 1: zamrzni backbone
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# ================= FAZA 1 – FEATURE EXTRACTOR =================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\n🚀 FAZA 1 – treniranje feature extractor-a\n")

history_feature = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FEATURE
)

# ================= FAZA 2 – FINE-TUNING =================
print("\n🔥 FAZA 2 – fine-tuning MobileNetV2\n")

# otključaj backbone
base_model.trainable = True

# zamrzni sve osim zadnjih 30 slojeva
for layer in base_model.layers[:-30]:
    layer.trainable = False

# OBAVEZNO recompiliranje
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE
)

# ================= SPREMANJE MODELA =================
model.save("fashion_mobilenetv2_finetuned.keras")

print("\n✅ Model uspješno treniran i spremljen!")
