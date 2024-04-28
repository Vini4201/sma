

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import string
import re
import textblob
from textblob import TextBlob
import os

from wordcloud import WordCloud, STOPWORDS
from wordcloud import ImageColorGenerator
import warnings
# %matplotlib inline

#Read the JSON generated from the CLI command above and create a pandas dataframe
tweets_df = pd.read_csv(r'/content/sample_data/Tweets.csv')

from google.colab import files

uploaded = files.upload()

import pandas as pd

# Assuming the uploaded file is 'Tweets.csv'
tweets_df = pd.read_csv('Tweets.csv')

tweets_df.head(5)

tweets_df.to_csv()

tweets_df.shape

tweets_df.head

tweets_df.info()

tweets_df.value_counts()

#Heat Map for missing Values
plt.figure(figsize=(17, 5))
sns.heatmap(tweets_df.isnull(), cbar=True, yticklabels=False)
plt.xlabel("Column_Name", size=14, weight="bold")
plt.title("Places of missing values in column",size=17)
plt.show()

import plotly.graph_objects as go
Top_Location_Of_tweet= tweets_df['airline'].value_counts().head (10)

print(Top_Location_Of_tweet)

from nltk. corpus import stopwords
stop = stopwords.words('english')
tweets_df['text'].apply(lambda x: [item for item in x if item not in stop])
tweets_df.shape

!pip install tweet-preprocessor

#Remove unnecessary characters
punct  =  ['%','/',':','\\','&amp','&',';','?']

def remove_punctuations(text):
  for punctuation in punct:
    text = text.replace(punctuation,'')
  return text

tweets_df['text'] = tweets_df['text'].apply(lambda x: remove_punctuations(x))

#Drop tweets that has empty text fields
tweets_df['text'].replace( '', np.nan, inplace=True)
tweets_df.dropna(subset=["text"],inplace=True)
len(tweets_df)

tweets_df = tweets_df.reset_index(drop=True)
tweets_df.head()

from sklearn.feature_extraction. text import TfidfVectorizer, CountVectorizer

# Commented out IPython magic to ensure Python compatibility.
sns.set_style('whitegrid')
# %matplotlib inline
stop=stop+['Virgin America' , 'San Francisco' , 'Boston' , 'New York', ' customer' ,'flight' , 'airline', 'San Diego' , 'Oakland' , 'California']
def plot_20_most_common_words(count_data, count_vectorizer) :
  import matplotlib. pyplot as plt
  words = count_vectorizer.get_feature_names()
  total_counts = np. zeros(len(words))
  for t in count_data:
    total_counts = t.toarray()[0]

  count_dict = (zip(words, total_counts))
  count_dict = sorted(count_dict, key=lambda x:x[1],reverse=True)[0:20]
  words = [w[0] for w in count_dict]
  counts = [w[1] for w in count_dict]
  x_pos = np.arange(len(words))

  plt.figure(2, (40,40))
  plt.subplot(title = '20 most common words')
  sns. set_context('notebook',font_scale=4,rc={ 'lines.linewidth' :2.5})
  sns.barplot(x_pos, counts, palette='husl')
  plt.xticks(x_pos, words, rotation=90)
  plt.xlabel('words')
  plt.ylabel('counts')
  plt.show()

count_vectorizer = CountVectorizer(stop_words=stop)
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(tweets_df['text'])
# print(count_vectorizer)
# print(count_data)
# Visualise the 20 most common words
plot_20_most_common_words(count_data,count_vectorizer)
plt.savefig('saved_figure.png')

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

sns.set_style('whitegrid')
# %matplotlib inline

stop = stop + ['Virgin America', 'San Francisco', 'Boston', 'New York', 'customer', 'flight', 'airline', 'San Diego', 'Oakland', 'California']

def plot_20_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names_out()
    total_counts = np.zeros(len(words))

    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = dict(zip(words, total_counts))
    count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)[:20]

    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]

    x_pos = np.arange(len(words))

    plt.figure(figsize=(12, 6))
    sns.set_context('notebook', font_scale=1.5)
    sns.barplot(x=x_pos, y=counts, palette='husl')
    plt.title('20 most common words')
    plt.xticks(x_pos, words, rotation=45, ha='right')
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.show()


count_vectorizer = CountVectorizer(stop_words=stop)
count_data = count_vectorizer.fit_transform(tweets_df['text'])

# Visualize the 20 most common words
plot_20_most_common_words(count_data, count_vectorizer)

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

def get_top_n_bigram(corpus, n=None) :
  vec = CountVectorizer(ngram_range=(2, 4), stop_words="english").fit(corpus)
  bag_of_words = vec.transform(corpus)
  sum_words = bag_of_words.sum(axis=0)
  words_freq =[(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
  words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
  return words_freq[:n]

common_words = get_top_n_bigram(tweets_df['text'] , 8)
mydict={}
for word, freq in common_words:
  bigram_df = pd.DataFrame(common_words,columns = ['ngram', 'count'])

bigram_df.groupby( 'ngram' ).sum()['count'].sort_values(ascending=False).sort_values().plot.barh(title = 'Top 8 bigrams',color='orange' , width=.4, figsize=(12,8),stacked = True)

def get_subjectivity(text):
  return TextBlob(text).sentiment.subjectivity
def get_polarity(text):
  return TextBlob(text).sentiment.polarity

tweets_df['subjectivity']=tweets_df[ 'text'].apply(get_subjectivity)
tweets_df['polarity']=tweets_df[ 'text'].apply(get_polarity)
tweets_df.loc[:,['airline','text','airline_sentiment','subjectivity','polarity']].head(20)

"""### Polarity Scores ###
Polarity scores are numerical values that range from -1 to 1, where -1 indicates a very negative sentiment, 0 indicates a neutral sentiment, and 1 indicates a very positive sentiment.
Polarity scores can help you quickly identify the overall mood of a text, whether it is a product review, a social media post, or a customer feedback.
Polarity scores may not capture the nuances and context of a text, such as sarcasm, irony, humor, or mixed emotions.
Polarity scores may not reflect the intensity or importance of a sentiment, such as how strongly or weakly a person feels about something

### Subjectivity Score ###
Subjectivity scores are numerical values that range from 0 to 1, where 0 indicates a very objective text, and 1 indicates a very subjective text.
Objective texts are based on facts, evidence, or logic, while subjective texts are based on opinions, feelings, or personal views.
Subjectivity scores can help you filter out irrelevant or biased texts, and focus on the ones that express genuine sentiments.

**5. Sentiment Analysis**
"""

tweets_df['textblob_score'] =tweets_df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

neutral_threshold=0.05

tweets_df['textblob_sentiment']=tweets_df[ 'textblob_score'].apply(lambda c:'positive' if c >= neutral_threshold else ('Negative' if c <= -(neutral_threshold) else 'Neutral' ) )

textblob_df =  tweets_df[['text','textblob_sentiment','retweet_count']]
textblob_df

textblob_df["textblob_sentiment"].value_counts()

textblob_df["textblob_sentiment"].value_counts().plot.barh(title = 'Sentiment Analysis',color='orange' , width=.4, figsize=(12,8),stacked = True)

df_positive=textblob_df[textblob_df['textblob_sentiment']=='positive' ]

df_very_positive=df_positive[df_positive['retweet_count']>0]

df_very_positive.head()

df_negative=textblob_df[textblob_df['textblob_sentiment']=='Negative' ]

df_negative

df_neutral=textblob_df[textblob_df['textblob_sentiment']=='Neutral' ]

df_neutral

"""**Create a Word Cloud**"""

from wordcloud import WordCloud, STOPWORDS
from PIL import Image

#Creating the text variable
positive_tw =" ".join(t for t in df_very_positive.text)
# Creating word _ cloud with text as argument in . generate() rtpthod
word_cloud1 = WordCloud(collocations = False, background_color = 'white') .generate(positive_tw)
# Display the generated Word Cloud
plt. imshow(word_cloud1, interpolation='bilinear')
plt.axis('off')
plt.show()

!pip install --upgrade pip

#!pip install --upgrade Pillow

!pip install wordcloud==1.8.0

#Creating the text variable
negative_tw =" ".join(t for t in df_negative.text)
# Creating word _ cloud with text as argument in . generate() rtpthod
word_cloud2 = WordCloud(collocations = False, background_color = 'white') .generate(negative_tw)
# Display the generated Word Cloud
plt. imshow(word_cloud2, interpolation='bilinear')
plt.axis('off')
plt.show()

#Creating the text variable
neutral_tw =" ".join(t for t in df_neutral.text)
# Creating word _ cloud with text as argument in . generate() rtpthod
word_cloud2 = WordCloud(collocations = False, background_color = 'white') .generate(neutral_tw)
# Display the generated Word Cloud
plt. imshow(word_cloud2, interpolation='bilinear')
plt.axis('off')
plt.show()