import tweepy

from secret_keys import secret_keys

CONSUMER_KEY = secret_keys["consumer_key"]
CONSUMER_SECRET = secret_keys["consumer_secret"]
ACCESS_TOKEN = secret_keys["access_token"]
ACCESS_SECRET = secret_keys["access_secret"]

def get_api():
    """ Sets up Twitter API """
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth)
    return api

if __name__ == "__main__":
    api = get_api()
    print(api)