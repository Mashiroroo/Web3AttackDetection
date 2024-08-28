import tweepy

# 设置    
consumer_key = "YOUR_CONSUMER_KEY"
consumer_secret = "YOUR_CONSUMER_SECRET"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"


# 创建 StreamListener 子类
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)


# 进行认证
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# 创建 Stream 对象
api = tweepy.API(auth)
myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)

# 监听特定账号的更新
target_accounts = ["account1", "account2"]
myStream.filter(follow=target_accounts)
