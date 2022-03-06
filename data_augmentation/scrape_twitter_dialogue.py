import os
import argparse
from typing import List

import tweepy

from secret_keys import secret_keys

CONSUMER_KEY = secret_keys["consumer_key"]
CONSUMER_SECRET = secret_keys["consumer_secret"]
ACCESS_TOKEN = secret_keys["access_token"]
ACCESS_SECRET = secret_keys["access_secret"]
BEARER_TOKEN = secret_keys["bearer_token"]

MMHS150K_DATA_PATH = "../data/MMHS150K/splits"
TRAIN_DATA = os.path.join(MMHS150K_DATA_PATH, "train_ids.txt")
TEST_DATA = os.path.join(MMHS150K_DATA_PATH, "test_ids.txt")
VAL_DATA = os.path.join(MMHS150K_DATA_PATH, "val_ids.txt")

def get_api():
    """ Sets up Twitter API """
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth)
    return api

def get_client():
    """ Sets up Twitter v2 API Client """
    client = tweepy.Client(BEARER_TOKEN)
    return client

def load_tweet_ids(split: str = "train"):
    """ Loads the tweet IDs from the MMHS150K data """
    filename = TRAIN_DATA
    if split == "test":
        filename = TEST_DATA
    elif split == "val":
        filename = VAL_DATA

    with open(filename, "r") as file:
        tweet_ids = [line.rstrip() for line in file]
    return tweet_ids

def scrape_tweet_replies(api: 'tweepy.api.API', client: 'tweepy.client.Client',
    tweet_ids: List[str], api_version: int = 1):
    """ Scrapes the tweet replies to the given list of tweet IDs """
    # Maps tweet ID to list of tweet reply Tweepy objects
    all_replies = dict()

    for indx, tweet_id in enumerate(tweet_ids):
        print(f"Trying to scrape replies to tweet {indx}...")

        if api_version == 2:
            # Twitter API v2, using conversation_id
            replies = []
            for tweet in client.search_recent_tweets(
                query="conversation_id:" + tweet_id
            ):
                replies.append(tweet)

            if len(replies) > 0:
                all_replies[tweet_id] = replies

        elif api_version == 1:
            # Twitter API v1, using Cursor search with in_reply_to_status_id
            try:
                tweet = api.get_status(tweet_id)._json
                user_name = tweet["user"]["screen_name"]

                replies = []
                for tweet in tweepy.Cursor(
                    api.search_tweets, q="to:" + user_name,
                    result_type="recent"
                ).items(1000):
                    if (hasattr(tweet, "in_reply_to_status_id_str")
                        and tweet.in_reply_to_status_id_str == tweet_id):
                        replies.append(tweet)

                if len(replies) > 0:
                    all_replies[tweet_id] = replies
            except:
                # If we try to access a tweet of a suspended user, Tweepy returns
                # the following error
                # tweepy.errors.Forbidden: 403 Forbidden 63 - User has been suspended.
                # In this case, we just continue to try the next week
                continue

    return all_replies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--twitter_api_version", type=int, default=1)
    args = parser.parse_args()

    # Twitter API v1
    api = get_api()
    print(api)

    # Twitter API v2 Client
    client = get_client()
    print(client)

    tweet_ids = load_tweet_ids()
    replies = scrape_tweet_replies(api, client, tweet_ids, api_version=args.twitter_api_version)
