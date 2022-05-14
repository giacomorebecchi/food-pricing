import os
import re
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()

_LOCATION_URL_REGEX = re.compile(
    r"(?:http[s]{0,1}://(?:[a-z]+\.(?:it|fr|en|es|de))(?:/(?:[a-zA-Z]{2,3}))?)?/restaurants/(?P<city>[a-zA-Z0-9\-\']*)/(?P<zone>[a-zA-Z0-9\-\']*)(?:/){0,1}"
)


def get_sitemap() -> str:
    url = os.environ["SITEMAP_URL"]
    response = requests.get(url)
    return response.text


def find_locations(text: str) -> List:
    matches = re.findall(_LOCATION_URL_REGEX, text)
    locations = list(dict.fromkeys(matches))
    return locations
