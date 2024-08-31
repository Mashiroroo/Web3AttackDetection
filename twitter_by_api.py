import tweepy
import requests
import os
from datetime import datetime
import time
import json

# Twitter API v2 凭据
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAKNcvgEAAAAAxEAgfug2gVih5qtPkDUTjU1vccI%3DhRERSJ6txXd23RuWWhcXN5XEyZiGbEdC5JkfqAKbboX0A5Ctpm'

# 监控的 Twitter 账号列表
usernames = [
    "CyversAlerts",
    "Cyvers_",
    "BlockSecTeam",
    "Phalcon_xyz",
    "SlowMist_Team",
    "PeckShieldAlert",
    "peckshield"
]

# 缓存用户ID的文件名
cache_file = "user_ids_cache.json"

# 检查是否有缓存的用户ID
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        user_ids = json.load(f)
else:
    user_ids = []

# 如果缓存为空或者用户列表有变化，重新获取用户ID
client = tweepy.Client(bearer_token=bearer_token)
if not user_ids or len(user_ids) != len(usernames):
    user_ids = []
    for username in usernames:
        while True:
            try:
                user = client.get_user(username=username)
                user_ids.append(user.data.id)
                break  # 成功获取ID后跳出循环
            except tweepy.errors.TooManyRequests:
                print(f"Rate limit exceeded for {username}. Waiting 15 minutes before retrying...")
                time.sleep(900)  # 等待15分钟后重试
            except Exception as e:
                print(f"An error occurred: {e}")
                break  # 处理其他可能的异常并跳出循环

    # 缓存用户ID到文件
    with open(cache_file, "w") as f:
        json.dump(user_ids, f)

# 创建存放图片的文件夹
if not os.path.exists("images"):
    os.makedirs("images")


# 继承并定义自定义的流处理器
class TweetStreamListener(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        if tweet.author_id in user_ids:
            print(f"New tweet from {tweet.author_id}: {tweet.text}")

            # 获取推文的详细信息，包括媒体信息
            tweet_details = client.get_tweet(tweet.id, expansions=["attachments.media_keys"], media_fields=["url"])
            if tweet_details.includes.get("media"):
                for media in tweet_details.includes["media"]:
                    if media["type"] == "photo":
                        img_url = media["url"]
                        print(f"Image found: {img_url}")
                        img_data = requests.get(img_url).content
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        with open(f"images/{tweet.id}_{timestamp}.jpg", "wb") as handler:
                            handler.write(img_data)
                            print(f"Image saved as images/{tweet.id}_{timestamp}.jpg")


# 初始化监听器
stream_listener = TweetStreamListener(bearer_token)

# 清除现有的规则以避免重复
try:
    existing_rules = stream_listener.get_rules()
    if existing_rules.data:
        rule_ids = [rule.id for rule in existing_rules.data]
        stream_listener.delete_rules(rule_ids)
except Exception as e:
    print(f"Error clearing rules: {e}")

# 分批添加规则以避免超过API限制
try:
    for i in range(0, len(user_ids), 5):
        batch_ids = user_ids[i:i + 5]
        stream_listener.add_rules(tweepy.StreamRule(value=f"from:{' OR from:'.join([str(id) for id in batch_ids])}"))
except tweepy.errors.Forbidden as e:
    print(f"Error adding rules: {e}")

# 开始监听推文流
try:
    stream_listener.filter(expansions=["author_id"])
except tweepy.errors.Forbidden as e:
    print(f"Error starting stream: {e}")
