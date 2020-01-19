import token

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import nltk
import re
from wordcloud import WordCloud
import time
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# Load spam email data from the csv file
emails_data = pd.read_csv("data/emails.csv")  # sep='\t', names=["text", "spam"]
# print(emails_data.head())

# Lets read a single email
# print(emails_data.at[58, 'text'])

# print(emails_data.describe())
# print(emails_data.shape)
# print(emails_data.groupby('spam').describe())
# data['length'] = data['text'].apply(len)
# print(data.head())

# Checking class distribution
# print(emails_data.groupby('spam').count())
# 23.88% emails are spam which seems good enough for our task

# print(emails_data.spam.value_counts())

# Plot a graph with the data
# label_count = emails_data.spam.value_counts()
# print(label_count)
# print(label_count.index)
#
# # plt.figure(figsize=(12, 6))
# plt.bar(label_count.index, data=label_count.values, height=label_count.values, align='center',
#         alpha=0.5)
#
# plt.xticks(label_count.index, label_count.values, rotation='vertical')
#
# plt.xlabel('Spam', fontsize=15)
# plt.ylabel('Counts', fontsize=15)
#
# plt.title('Spam distribution chart')
#
# plt.show()

# Lets check if email length is coorelated to spam/ham
emails_data['length'] = emails_data['text'].map(lambda text: len(text))

# print(emails_data.groupby('spam').length.describe())

# emails length have some extreme outliers, lets set a length threshold & check length distribution
emails_subset = emails_data[emails_data.length < 1800]
emails_subset.hist(column='length', by='spam', bins=50)

# Nothing much here, lets process the contents of mail now for building spam filter


# Text data processing
emails_data['tokens'] = emails_data['text'].map(lambda text: nltk.tokenize.word_tokenize(text))

# print(emails_data['tokens'][1])

# Stop Words Removal
# Stop words usually refers to the most common words in a language like 'the', 'a', 'as' etc.
# These words usually do not convey any useful information needed for spam filter so lets remove them.

# Removing stop words

stop_words = set(nltk.corpus.stopwords.words('english'))
# print(stop_words)

emails_data['filtered_text'] = emails_data['tokens'].map(
    lambda tokens: [w for w in tokens if not w in stop_words])

# Every mail starts with 'Subject :' lets remove this from each mail

emails_data['filtered_text'] = emails_data['filtered_text'].map(lambda text: text[2:])

# Lets compare an email with stop words removed

# print(emails_data['tokens'][3], end='\n\n')
# print(emails_data['filtered_text'][3])

# Mails still have many special charater tokens which may not be relevant for spam filter, lets remove these
# Joining all tokens together in a string
emails_data['filtered_text'] = emails_data['filtered_text'].map(lambda text: ' '.join(text))

# removing special characters from each mail
emails_data['filtered_text'] = emails_data['filtered_text'].map(
    lambda text: re.sub('[^A-Za-z0-9]+', ' ', text))

# Lemmatization
#   Its the process of grouping together the inflected forms of a word
#   so they can be analysed as a single item, identified by the word's lemma, or dictionary form.
#   so word like 'moved' & 'moving' will be reduced to 'move'.

wnl = nltk.WordNetLemmatizer()
emails_data['filtered_text'] = emails_data['filtered_text'].map(lambda text: wnl.lemmatize(text))

# Lets check one of the mail again after all these preprocessing steps
# print(emails_data['filtered_text'][4])

# Wordcloud of spam mails
# spam_words = ''.join(list(emails_data[emails_data['spam'] == 1]['filtered_text']))
# spam_word_cloud = WordCloud(width=512, height=512).generate(spam_words)
# plt.figure(figsize=(10, 8), facecolor='k')
# plt.imshow(spam_word_cloud)
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.show()

# Wordcloud of non-spam mails
# spam_words = ''.join(list(emails_data[emails_data['spam'] == 0]['filtered_text']))
# spam_word_cloud = WordCloud(width=512, height=512).generate(spam_words)
# plt.figure(figsize=(10, 8), facecolor='k')
# plt.imshow(spam_word_cloud)
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.show()

# Spam Filtering Models
# After preprocessing we have clean enough text,
# lets convert these mails into vectors of numbers using 2 popular methods: Bag of Words & TF-IDF.
# After getting vectors for each mail we will build our classifier using Naive Bayes.

# Bag of Words
count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(emails_data['filtered_text'].values)
print(counts.shape)

# Naive Bayes Classifier
classifier = MultinomialNB()
targets = emails_data['spam'].values
classifier.fit(counts, targets)

# Predictions on sample text
examples = ['cheap Viagra', "Forwarding you minutes of meeting"]
example_counts = count_vectorizer.transform(examples)
predictions = classifier.predict(example_counts)

print(predictions)

# 2. TF-IDF

tfidf_vectorizer = TfidfTransformer().fit(counts)
tfidf = tfidf_vectorizer.transform(counts)

print(tfidf.shape)

# Naive Bayes Classifier
classifier = MultinomialNB()
targets = emails_data['spam'].values
classifier.fit(counts, targets)

# Predictions on sample text
examples = ['Free Offer Buy now', "Lottery from Nigeria", "Please send the files"]
example_counts = count_vectorizer.transform(examples)
example_tfidf = tfidf_vectorizer.transform(example_counts)
predictions_tfidf = classifier.predict(example_tfidf)

print(predictions_tfidf)
