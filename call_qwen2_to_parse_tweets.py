import requests

url = 'http://219.145.11.116:50000/generate'
data = {
    "prompt": "Give me a short introduction to large language model."
}

response = requests.post(url, json=data)
print(response.status_code)
print(response.json())
