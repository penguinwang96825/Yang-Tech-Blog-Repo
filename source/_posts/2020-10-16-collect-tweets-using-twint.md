---
title: Collect Tweets using Twint
top: false
cover: false
toc: true
mathjax: true
date: 2020-10-16 22:41:28
img: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2020/10/16/2020-10-16-collect-tweets-using-twint/wallhaven-j3ej9y.jpg?raw=true
coverImg: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2020/10/16/2020-10-16-collect-tweets-using-twint/wallhaven-j3ej9y.jpg?raw=true
summary: Twint is a Python-based advanced Twitter scraping app that allows you to scrape Tweets from Twitter profiles without having to use Twitter's API. Twint makes use of Twitter's search operators to allow you to scrape Tweets from specific individuals, scrape Tweets referring to specific themes, hashtags, and trends, and sort out sensitive information like e-mail and phone numbers from Tweets. This is something I find quite handy, and you can get fairly creative with it as well.
categories: Data Science
tags:
	- Python
	- Twitter
	- Tweet
---

# Introduction

[Twint](https://github.com/twintproject/twint) is a Python-based advanced Twitter scraping app that allows you to scrape Tweets from Twitter profiles without having to use Twitter's API. Twint makes use of Twitter's search operators to allow you to scrape Tweets from specific individuals, scrape Tweets referring to specific themes, hashtags, and trends, and sort out sensitive information like e-mail and phone numbers from Tweets. This is something I find quite handy, and you can get fairly creative with it as well.

# Installation

You could find it difficult to install Twint for some reason, therefore I'll explain you how to do so in the steps below.

1. `pip install --upgrade aiohttp_socks`
2. `pip3 install --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint`
3. In orser to solve `OSError: [WinError 87] The parameter is incorrect.` please open `output.py` in `./src/twint/twint` first. Make the following changes to fixes this issue:

- Add (line#9):
```python
import string
``` 

- Replace (line#123):
```python
word = ''
for i in output:
	if i in string.printable:
		word = word + i
print(word.replace('\n', ' '))
```

# Easy Example

Twint now allows custom formatting and can be used as a module. More information can be found over [here](https://github.com/twintproject/twint/wiki).

```python
import twint

# Configure
c = twint.Config()
c.Username = "Bitcoin"
c.Search = "great"

# Run
twint.run.Search(c)
```

The other way is to scrape the tweets through command line.

```bash
twint -u Bitcoin --csv --output tweets.csv --since 2014-01-01 
```

## CLI Basic Examples and Combos
A few simple examples to help you understand the basics:

- `twint -u username` - Scrape all the Tweets of a *user* (doesn't include **retweets** but includes **replies**).
- `twint -u username -s pineapple` - Scrape all Tweets from the *user*'s timeline containing _pineapple_.
- `twint -s pineapple` - Collect every Tweet containing *pineapple* from everyone's Tweets.
- `twint -u username --year 2014` - Collect Tweets that were tweeted **before** 2014.
- `twint -u username --since "2015-12-20 20:30:15"` - Collect Tweets that were tweeted since 2015-12-20 20:30:15.
- `twint -u username --since 2015-12-20` - Collect Tweets that were tweeted since 2015-12-20 00:00:00.
- `twint -u username -o file.txt` - Scrape Tweets and save to file.txt.
- `twint -u username -o file.csv --csv` - Scrape Tweets and save as a csv file.
- `twint -u username --email --phone` - Show Tweets that might have phone numbers or email addresses.
- `twint -s "Donald Trump" --verified` - Display Tweets by verified users that Tweeted about Donald Trump.
- `twint -g="48.880048,2.385939,1km" -o file.csv --csv` - Scrape Tweets from a radius of 1km around a place in Paris and export them to a csv file.
- `twint -u username -es localhost:9200` - Output Tweets to Elasticsearch
- `twint -u username -o file.json --json` - Scrape Tweets and save as a json file.
- `twint -u username --database tweets.db` - Save Tweets to a SQLite database.
- `twint -u username --followers` - Scrape a Twitter user's followers.
- `twint -u username --following` - Scrape who a Twitter user follows.
- `twint -u username --favorites` - Collect all the Tweets a user has favorited (gathers ~3200 tweet).
- `twint -u username --following --user-full` - Collect full user information a person follows
- `twint -u username --timeline` - Use an effective method to gather Tweets from a user's profile (Gathers ~3200 Tweets, including **retweets** & **replies**).
- `twint -u username --retweets` - Use a quick method to gather the last 900 Tweets (that includes retweets) from a user's profile.
- `twint -u username --resume resume_file.txt` - Resume a search starting from the last saved scroll-id.

More detail about the commands and options are located in the [wiki](https://github.com/twintproject/twint/wiki/Commands)

# Conclusion

There are several benefits of using Twint. First, it can fetch all tweets, and Twitter API limits to last 3200 tweets only. Second, it can be used anonymously and without Twitter Developer sign up. Finally, it can be fast initial setup and no rate limitations. There are a lot more search features to play with within Twint, you definitely want to play with it by yourself!

# References

1. https://medium.com/analytics-vidhya/how-to-scrape-tweets-from-twitter-with-python-twint-83b4c70c5536
2. https://github.com/twintproject/twint