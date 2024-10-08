import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

url = 'https://www.4byte.directory/api/v1/signatures/'
signatures = []
page = 1
# proxies = {
#     "http": "http://127.0.0.1:7890",
#     "https": "http://127.0.0.1:7890",
# }

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

while True:
    try:
        response = session.get(url, params={'page': page}, timeout=10)
        data = response.json()
        signatures.extend(data['results'])
        if not data['next']:
            break
        page += 1
    except requests.exceptions.Timeout:
        print(f"Request timed out on page {page}. Retrying...")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        break

with open('4byte_signatures.json', 'w') as f:
    json.dump(signatures, f)
