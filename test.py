import requests

def get_coordinates(city_name):
    """
    Dohvati latitude i longitude za ime grada koristeći Open-Meteo Geocoding API.
    """
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&format=json"
    response = requests.get(url)
    data = response.json()

    if "results" not in data or len(data["results"]) == 0:
        raise ValueError(f"Grad '{city_name}' nije pronađen.")

    result = data["results"][0]
    return result["latitude"], result["longitude"], result.get("timezone", "auto")


def get_current_temperature(city_name):
    """
    Dohvati trenutnu temperaturu za zadani grad.
    """
    lat, lon, timezone = get_coordinates(city_name)
    
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true&temperature_unit=celsius&timezone={timezone}"
    )

    response = requests.get(url)
    data = response.json()

    if "current_weather" not in data:
        raise ValueError("Ne mogu dohvatiti trenutnu temperaturu.")

    temp = data["current_weather"]["temperature"]
    return temp


if __name__ == "__main__":
    city = input("Unesi ime grada: ")
    try:
        temperature = get_current_temperature(city)
        print(f"Trenutna temperatura u {city} je {temperature}°C.")
    except Exception as e:
        print("Greška:", e)
