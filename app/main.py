from fastapi import FastAPI, UploadFile, File
import shutil
import os

from .classifier import classify_image
from .color_extractor import extract_color

app = FastAPI(title="AI Clothing Service")

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_clothing(file: UploadFile = File(...)):
    image_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    category = classify_image(image_path)
    color = extract_color(image_path)

    os.remove(image_path)

    return {
        "category": category,
        "color": color
    }
