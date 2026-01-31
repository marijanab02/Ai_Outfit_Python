from fastapi import FastAPI, UploadFile, File
import shutil
import os
from typing import List
from pydantic import BaseModel
from .weather import get_current_temperature
from .outfit_builder import build_outfit

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


class ClothingItem(BaseModel):
    id: int
    category: str
    color: str
    image_url: str

class OutfitRequest(BaseModel):
    city: str
    country: str | None = None
    clothes: List[ClothingItem]

    
@app.post("/suggest-outfit")
def suggest_outfit(req: OutfitRequest):
    print("CITY:", req.city)

    temperature = get_current_temperature(req.city)

    print("TEMPERATURE:", temperature)

    outfit = build_outfit(
        [c.dict() for c in req.clothes],
        temperature
    )

    if not outfit:
        return {
            "error": "No suitable outfit",
            "temperature": temperature
        }

    return {
        "temperature": temperature,
        "outfit": outfit
    }
