from math import sqrt
from typing import Tuple

from geopy import Location, Nominatim
from geopy.extra.rate_limiter import RateLimiter
from src.data.cache import Cache

geocoder = Nominatim(user_agent="food-pricing")
geocode = RateLimiter(geocoder.geocode, min_delay_seconds=1)


@Cache
def geocode_address(address: str) -> Tuple[float]:
    try:
        location: Location = geocode(address)
    except Exception as e:
        print(e)
        return None
    if location:  # if a match was found
        return location.latitude, location.longitude
    else:
        return None


def get_address(*args: str) -> str:
    address_list = [word for arg in args for word in arg.split("-")]
    return " ".join(list(dict.fromkeys(address_list)))


def distance(a: Tuple[float], b: Tuple[float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_coords(city: str, zone: str) -> Tuple[float]:
    address = get_address(city, zone)
    coords = geocode_address(address)
    if coords:
        return coords
    else:
        coords_city = geocode_address(get_address(city))
        coords_zone = geocode_address(get_address(zone))
        if coords_city and coords_zone and distance(coords_city, coords_zone) < 0.2:
            return coords_zone
        else:
            return coords_city
