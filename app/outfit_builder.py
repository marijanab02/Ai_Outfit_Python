import random

def build_outfit(clothes, temperature):
    allowed_categories = temperature_rules(temperature)

    filtered = [
        c for c in clothes
        if c["category"] in allowed_categories
    ]

    tops = [c for c in filtered if c["category"] in [
        "long_sleeve_top", "short_sleeve_top", "vest"
    ]]

    bottoms = [c for c in filtered if c["category"] in [
        "trousers", "skirt", "shorts"
    ]]

    jackets = [c for c in filtered if c["category"] == "jackets"]

    if not tops or not bottoms:
        return None

    top = random.choice(tops)
    bottom = random.choice(bottoms)

    if not compatible(
        color_group(top["color"]),
        color_group(bottom["color"])
    ):
        return None

    outfit = {
        "top": top,
        "bottom": bottom
    }

    if temperature < 15 and jackets:
        outfit["jacket"] = random.choice(jackets)

    return outfit

def temperature_rules(temp):
    if temp < 10:
        return ["jackets", "long_sleeve_top", "trousers", "vest"]
    elif temp < 18:
        return ["long_sleeve_top", "trousers", "skirt"]
    elif temp < 25:
        return ["short_sleeve_top", "skirt", "shorts"]
    else:
        return ["short_sleeve_top", "shorts", "skirt"]

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def color_group(hex_color):
    r, g, b = hex_to_rgb(hex_color)

    if abs(r - g) < 20 and abs(g - b) < 20:
        return "neutral"
    if r > b:
        return "warm"
    return "cool"


def compatible(c1, c2):
    if c1 == "neutral" or c2 == "neutral":
        return True
    return c1 != c2
