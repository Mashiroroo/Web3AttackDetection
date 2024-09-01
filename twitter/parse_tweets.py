import requests

url = 'http://192.168.1.200:50822/generate'
payload = "ALERT! Please revoke approvals to 0xe3a0bc3483ae5a04db7ef2954315133a6f7d228e ASAP!\n\nSubscribe to Phalcon to monitor attacks and take automatic actions: https://t.co/pN2xDiTk64"

data = {
    "prompt": f"这是一个区块链安全的报警推文。{payload}请告诉我上面有没有提到64位的攻击交易哈希？如果有的话，请返回推文中提到的64位的交易哈希。如果没有的话，给我返回'None'"
}

response = requests.post(url, json=data, timeout=10, proxies={"http": None, "https": None})
# print(response.status_code)
print(response.json())
