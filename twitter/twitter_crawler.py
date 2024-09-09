import asyncio
import random
import ssl
import httpx
from tqdm import tqdm
from twikit import Client
from parse_tweets import parse_tweets


# 处理新推文
def process_new_tweets(new_tweets):
    for tweet in new_tweets:
        parse_tweets(tweet)


async def fetch_tweets():
    client = Client('en-US', timeout=120, proxy='http://127.0.0.1:7890')

    # 登录
    await client.login(auth_info_1='shiro050822', auth_info_2='liuxingchen@xayytech.com', password='041129abc')

    client.save_cookies('cookies.json')
    client.load_cookies(path='./cookies.json')

    users_to_monitor = ['CyversAlerts', 'Cyvers_', 'BlockSecTeam', 'Phalcon_xyz', 'SlowMist_Team', 'PeckShieldAlert',
                        'peckshield', 'lunaray_co', 'ChainAegis', 'FortaNetwork', 'HashDit', '0xNickLFranklin',
                        'blockaid_', 'EXVULSEC', 'MetaTrustAlert', 'shoucccc', 'realScamSniffer', 'DecurityHQ',
                        'zachxbt', 'spreekaway', 'bbbb', 'hexagate_', 'CertiKAlert', 'MistTrack_io', 'AnciliaInc',
                        'BeosinAlert']
    last_tweet_id = {}  # 存储每个用户的最新推文ID

    # 初始化时，获取每个用户的最新一条推文ID
    print("Initializing...")
    for username in tqdm(users_to_monitor, desc="Initializing", ncols=100):
        try:
            user = await client.get_user_by_screen_name(username)
            tweets = await user.get_tweets('Tweets', count=1)  # 仅获取最新1条推文
            if tweets:
                last_tweet_id[username] = tweets[0].id  # 记录最新推文ID
        except (httpx.ConnectError, httpx.ReadTimeout, ssl.SSLWantReadError) as e:
            print(f"Error initializing tweets for {username}: {e}")
        await asyncio.sleep(random.randint(30, 60))

    while True:
        for username in users_to_monitor:
            try:
                try:
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
                        process_new_tweets(new_tweets)

                        print(f"Fetched {len(new_tweets)} new tweets from {username}")
                    else:
                        print(f"No new tweets from {username}")

                except (httpx.ConnectError, httpx.ReadTimeout, ssl.SSLWantReadError) as e:
                    print(f"Error fetching tweets for {username}: {e}")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        print(f"Rate limit exceeded. Sleeping for 30 minutes...")
                        await asyncio.sleep(1800)  # 休眠30分钟
            except Exception as e:
                print(e)
            await asyncio.sleep(random.randint(30, 60))

        # # 等待 30分钟与60分钟间一随机时间
        # wait_time = random.randint(1800, 3600)
        # await asyncio.sleep(wait_time)


# 在事件循环中运行fetch_tweets函数
asyncio.run(fetch_tweets())
