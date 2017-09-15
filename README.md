
# Twitter Sentiment

Can we use datasets available online to train a useful twitter sentiment analyzer?

## Data

- Twitter
  - [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
  	- 14,873 tweets
  	- Columns: tweet_id, airline_sentiment, airline_sentiment_confidence, negativereason, negativereason_confidence, airline, airline_sentiment_gold, name, negativereason_gold, retweet_count, text, tweet_coord, tweet_created, tweet_location, user_timezone
  	- Assumed location: `data/airline-tweets.csv`
  - [First GOP Debate Twitter Sentiment](https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment)
  	- 16,655 tweets
  	- Columns: id,candidate, candidate_confidence, relevant_yn, relevant_yn_confidence, sentiment, sentiment_confidence, subject_matter, subject_matter_confidence, candidate_gold,name, relevant_yn_gold, retweet_count, sentiment_gold, subject_matter_gold, text, tweet_coord, tweet_created, tweet_id, tweet_location, user_timezone
  	- Assumed location: `data/gop-debate-sentiment.csv`
  - [Combined dataset of  ](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/)
  	- 1,578,628 tweets
  	- Columns: ItemID, Sentiment, SentimentSource, SentimentText
  	- Assumed location: `data/combined-sentiment-dataset.csv`
