import asyncio
import random
from twikit import Client
import json
import time


# 定义一个函数来处理新推文
def process_new_tweets(username, new_tweets):
    # 在这里处理新推文，例如打印或执行其他操作
    print(f"Processing {len(new_tweets)} new tweets from {username}:")
    for tweet in new_tweets:
        print(tweet['full_text'])


async def fetch_tweets():
    client = Client('en-US', timeout=120, proxy='http://127.0.0.1:7890')

    # 登录
    await client.login(auth_info_1='shiro050822', auth_info_2='liuxingchen@xayytech.com', password='73942lxc')

    # 加载cookies
    client.load_cookies(path='cookies.json')

    # 监控的用户
    users_to_monitor = ['CyversAlerts', 'Cyvers_', 'BlockSecTeam', 'Phalcon_xyz', 'SlowMist_Team', 'PeckShieldAlert',
                        'peckshield', 'shiro050822']
    last_tweet_id = {}  # 用于存储每个用户的最新推文ID

    # 初始化时，获取每个用户的最新一条推文ID
    for username in users_to_monitor:
        user = await client.get_user_by_screen_name(username)
        tweets = await user.get_tweets('Tweets', count=1)  # 仅获取最新1条推文
        if tweets:
            last_tweet_id[username] = tweets[0].id  # 记录最新推文ID

    while True:
        for username in users_to_monitor:
            user = await client.get_user_by_screen_name(username)
            tweets = await user.get_tweets('Tweets', count=5)  # 获取最新5条推文

            new_tweets = []
            for tweet in reversed(tweets):  # 反向遍历推文，从最旧到最新
                if username not in last_tweet_id or tweet.id > last_tweet_id[username]:
                    new_tweets.append({
                        'created_at': tweet.created_at,
                        'favorite_count': tweet.favorite_count,
                        'full_text': tweet.full_text,
                    })
                    last_tweet_id[username] = tweet.id

            if new_tweets:
                # 调用处理新推文的函数
                process_new_tweets(username, new_tweets)

                # 将新推文保存为JSON格式
                with open(f'/data/{username}_new_tweets.json', 'w', encoding='utf-8') as f:
                    json.dump(new_tweets, f, ensure_ascii=False, indent=4)

                print(f"Fetched {len(new_tweets)} new tweets from {username}")
            else:
                print(f"No new tweets from {username}")

        # 等待 10分钟与30分钟间一随机时间
        wait_time = random.randint(600, 1800)
        time.sleep(wait_time)


# 在事件循环中运行fetch_tweets函数
asyncio.run(fetch_tweets())
