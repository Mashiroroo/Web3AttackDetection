import tweepy


def get_tweets(username, num_tweets):
    auth = tweepy.OAuthHandler('xF2s6ocad8qIWO5bhtcwxQAAN', 'RS0VrYCwhqY5jXrACLIb7ppPLRNjYCnD5wVsAmcWyBwiq5CmXV')
    auth.set_access_token('1827678027855085568-J5LxYCb7RK4MgmOGLWa8eau3diHWXA',
                          'PMtfPBjSOhVkt5Q4IIK7zNpcGLrxMSeJLMCuGQsVeKZGr')

    api = tweepy.API(auth)

    tweets = api.user_timeline(screen_name=username, count=num_tweets, tweet_mode='extended')

    for tweet in tweets:
        print(f"{tweet.user.screen_name}\n{tweet.full_text}\n{'-' * 50}")


if __name__ == "__main__":
    user = '@CyversAlerts'  # Replace with the desired Twitter username
    num_tweets = 10  # The number of tweets you want to retrieve
    get_tweets(user, num_tweets)
