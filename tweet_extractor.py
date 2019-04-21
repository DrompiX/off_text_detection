import csv
import sys
from twython import Twython
from urllib import request
from bs4 import BeautifulSoup

csv.field_size_limit(sys.maxsize)


def read_data(path):
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
    read_data(path)


main()