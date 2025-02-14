from httpx import get, Response
import json

from setup_env import WEATHERAPI_API_KEY


def get_weather_data(location: str) -> str:
    endpoint: str = f"https://api.weatherapi.com/v1/current.json?key={WEATHERAPI_API_KEY}&q={location}"
    response: Response = get(endpoint)
    return json.dumps(response.json(), indent=2)


if __name__ == "__main__":
    print(get_weather_data("Taipai City"))
