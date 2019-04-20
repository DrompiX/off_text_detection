import csv
import sys
from twython import Twython
from urllib import request
from bs4 import BeautifulSoup

csv.field_size_limit(sys.maxsize)


# def get_twitter_access():
#     CONSUMER_KEY = "<consumer key>"
#     CONSUMER_SECRET = "<consumer secret>"
#     OAUTH_TOKEN = "<application key>"
#     OAUTH_TOKEN_SECRET = "<application secret"
#     twitter = Twython(
#         CONSUMER_KEY, CONSUMER_SECRET,
#         OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

#     return twitter
    # tweet = twitter.show_status(id=id_of_tweet)
    # print(tweet['text'])

def read_data(path):
    # twitter = get_twitter_access()
    processed = 0
    with open("data.nosync/tweets.csv", "w+") as tf:
        writer = csv.writer(tf, delimiter=',')
        with open(path) as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            tweets_n = 16907
            for t_id, t_class in rows:
                try:
                    feed = request.urlopen(f"https://twitter.com/anyuser/status/{t_id}").read()
                    soup = BeautifulSoup(feed, features='html.parser')
                    tweet = soup.find("p", {"class": "TweetTextSize TweetTextSize--jumbo js-tweet-text tweet-text"})
                    if tweet:
                        writer.writerow([t_id, tweet.get_text(), t_class])
                except KeyboardInterrupt:
                    sys.exit()
                except:
                    pass

                processed += 1

                sys.stdout.write(f'\rTweets processed = {processed}/{tweets_n}')
                sys.stdout.flush()
        

def main():
    path = "data.nosync/id2class.csv"
    # print(request.urlopen("https://twitter.com/anyuser/status/572342978255048705").read())
    # id2class = read_data(path)
    read_data(path)


main()