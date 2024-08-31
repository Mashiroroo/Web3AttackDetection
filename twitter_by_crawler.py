import requests
from bs4 import BeautifulSoup
import time

# 替换为目标用户的用户名
username = 'BlockSecTeam'
url = f'https://twitter.com/{username}'


def check_updates():
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找推文的容器（根据实际的 HTML 结构调整）
    tweets = soup.find_all('article')

    for tweet in tweets:
        tweet_content = tweet.get_text()
        print(tweet_content)


# 定期检查更新
while True:
    check_updates()
    time.sleep(300)  # 每5分钟检查一次
