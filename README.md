
# Twitter Sentiment

Can we use datasets available online to train a useful twitter sentiment analyzer?

## Usage

First, download the datasets in the Data section, and move them to the `./data` directory with the expected file names.  Symlinking is fine (and is generally a good idea).

Prepping data:

```
$ ls data
airline-tweets.csv              gop-debate-sentiment.csv  combined-sentiment-dataset.csv
$ python3 main.py prep
Reading data...
1607125it [00:04, 385203.15it/s]
Partitioning data...
100%|█████████████████████████████████████████████████████████████████| 1607125/1607125 [00:00<00:00, 1974808.78it/s]
Writing heldout data to data/text_data_heldout.csv...
100%|████████████████████████████████████████████████████████████████████| 161272/161272 [00:00<00:00, 294697.59it/s]
Writing test data to data/text_data_test.csv...
100%|████████████████████████████████████████████████████████████████████| 321422/321422 [00:01<00:00, 292592.31it/s]
Writing train data to data/text_data_train.csv...
100%|██████████████████████████████████████████████████████████████████| 1124431/1124431 [00:03<00:00, 297164.11it/s]
$ ls data
airline-tweets.csv              gop-debate-sentiment.csv  text_data_train_test.csv
combined-sentiment-dataset.csv  text_data_heldout.csv

```

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


## Procecssed Format

Data is turned into a set of CSVs:

- `data/text_data_<train/test/heldout>.csv`
  - CSV with text (with data source/category prepended) and classifications
  - Heldout data should be used rarely!

