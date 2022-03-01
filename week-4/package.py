import pandas as pd
import nltk
import re
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweets(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)

    tokenizer = TweetTokenizer(
        preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean


def get_dict(file_name):
    my_file = pd.read_csv(file_name, delimiter=' ')
    file = {}
    for i in range(len(my_file)):
        en = my_file.iloc[i][0]
        fr = my_file.iloc[i][1]
        file[en] = fr

    return file


def cosine_similarity(A, B):
    axb = np.dot(A, B)
    a_norm = np.linalg.norm(A)
    b_norm = np.linalg.norm(B)
    cos_b = axb/(a_norm * b_norm)
    return cos_b
