import requests

def get_coordinates(city):
    url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={city}&count=1&format=json"
    )
    data = requests.get(url).json()

    if "results" not in data:
        raise Exception("City not found")

    r = data["results"][0]
    return r["latitude"], r["longitude"], r.get("timezone", "auto")


def get_current_temperature(city):
    lat, lon, timezone = get_coordinates(city)

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true&temperature_unit=celsius"
        f"&timezone={timezone}"
    )

    data = requests.get(url).json()
    return data["current_weather"]["temperature"]
