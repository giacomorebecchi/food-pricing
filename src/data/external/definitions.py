import re

LOCATION_URL_REGEX = re.compile(
    r"(?:http[s]{0,1}://(?:[a-z]+\.(?:it|fr|en|es|de))(?:/(?:[a-zA-Z]{2,3}))?)?/restaurants/(?P<city>[a-zA-Z0-9\-\']*)/(?P<zone>[a-zA-Z0-9\-\']*)(?:/){0,1}"
)
