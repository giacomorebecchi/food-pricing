import os
import re
from typing import List

import requests
from dotenv import load_dotenv

from .definitions import LOCATION_URL_REGEX

load_dotenv()


def get_sitemap() -> str:
    url = os.environ["SITEMAP_URL"]
    response = requests.get(url)
    return response.text


def find_locations(text: str) -> List:
    matches = re.findall(LOCATION_URL_REGEX, text)
    locations = list(dict.fromkeys(matches))
    return locations
