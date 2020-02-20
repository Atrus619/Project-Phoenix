import requests
import urllib.parse
from config import Config as cfg


def get_latitude_and_longitude(location, company=None):
    if company:
        search_str = company + ', ' + location
    else:
        search_str = location

    request_url = build_geocoding_url(search_str)
    response = requests.get(request_url)

    resp_json_payload = response.json()
    coordinates = resp_json_payload['results'][0]['geometry']['location']
    return coordinates['lat'], coordinates['lng']


def build_geocoding_url(location):
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json'
    key = cfg.GCP_API_KEY
    address = urllib.parse.quote_plus(location)
    return f'{base_url}?key={key}&address={address}'
