import requests

url = 'http://192.168.1.200:50822/generate'
data = {
    "prompt": "Give me a short introduction to large language model."
}

response = requests.post(url, json=data, timeout=10, proxies={"http": None, "https": None})
# print(response.status_code)
print(response.json())
