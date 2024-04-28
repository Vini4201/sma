

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
tweets_df = pd.read_csv(r'/content/Tweets.csv')

# Or try this code to upload on colab
from google.colab import files

uploaded = files.upload()

tweets_df.head()

tweets_df.shape

tweets_df.head()

tweets_df.info()

tweets_df.value_counts(tweets_df['airline'])

tweets_df.value_counts(tweets_df['airline_sentiment_gold'])

tweets_df['airline_sentiment_gold'].isnull().sum()

tweets_df.value_counts()

tweets_df.isnull().sum()

#Heat Map for missing Values
plt.figure(figsize=(17, 5))
sns.heatmap(tweets_df.isnull(), cbar=True, yticklabels=False)
plt.xlabel("Column_Name", size=14, weight="bold")
plt.title("Places of missing values in column",size=17)
plt.show()

tweets_df.isnull().sum().plot(kind="bar") # plotting null valu count for various columns

import plotly.graph_objects as go
Top_Location_Of_tweet= tweets_df['airline'].value_counts().head (10)

print(Top_Location_Of_tweet)

from nltk. corpus import stopwords
stop = stopwords.words('english')
tweets_df['text'].apply(lambda x: [item for item in x if item not in stop])
tweets_df.shape

tweets_df['text'].head(10)

!pip install tweet-preprocessor

#Remove unnecessary characters
punct  =  ['%','/',':','\\','&amp','&',';','?']

def remove_punctuations(text):
  for punctuation in punct:
    text = text.replace(punctuation,'')
  return text

tweets_df['text'] = tweets_df['text'].apply(lambda x: remove_punctuations(x))

tweets_df['text'].head(10)

tweets_df['text'].isnull().sum()

#Drop tweets that has empty text fields
tweets_df['text'].replace( '', np.nan, inplace=True)
tweets_df.dropna(subset=["text"],inplace=True)
len(tweets_df)

tweets_df = tweets_df.reset_index(drop=True)
tweets_df.head()

from sklearn.feature_extraction. text import TfidfVectorizer, CountVectorizer

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



"""### **Use the data we have scraped in exp 2 (APSIT reviews), and clean the same**"""

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
tweets_df = pd.read_csv(r'/content/apsit.csv')

# Or try this code to upload on colab
from google.colab import files

uploaded = files.upload()

tweets_df.head(5)

tweets_df.shape

tweets_df.head()

tweets_df.info()

tweets_df.value_counts(tweets_df['Reviewer Name'])

tweets_df.value_counts(tweets_df['Review'])

tweets_df['Review'].isnull().sum()

tweets_df.value_counts()

tweets_df.isnull().sum()

#Heat Map for missing Values
plt.figure(figsize=(17, 5))
sns.heatmap(tweets_df.isnull(), cbar=True, yticklabels=False)
plt.xlabel("Column_Name", size=14, weight="bold")
plt.title("Places of missing values in column",size=17)
plt.show()

tweets_df.isnull().sum().plot(kind="bar") # plotting null valu count for various columns

import plotly.graph_objects as go
Top_Review= tweets_df['Review'].value_counts().head (10)

print(Top_Review)

from nltk.corpus import stopwords
stop = stopwords.words('english')
tweets_df['Review'] = tweets_df['Review'].apply(lambda x: [item for item in str(x).split() if item.lower() not in stop])
# Now, if you want to check the shape of your DataFrame
print(tweets_df.shape)

tweets_df['Review'].head(10)

!pip install tweet-preprocessor

import string

def remove_punctuations(review_list):
    # Check if the input is a list
    if isinstance(review_list, list):
        # Join the list elements into a single string
        review_str = ' '.join(map(str, review_list))
        # Remove punctuation
        review_no_punct = ''.join([char for char in review_str if char not in string.punctuation])
        return review_no_punct.split()  # Split the string back into a list
    else:
        return review_list

tweets_df['Review'] = tweets_df['Review'].apply(lambda x: remove_punctuations(x))

tweets_df['Review'].head(10)

tweets_df['Review'].isnull().sum()

#Drop tweets that has empty text fields
tweets_df['Review'].replace( '', np.nan, inplace=True)
tweets_df.dropna(subset=["Review"],inplace=True)
len(tweets_df)

tweets_df['Review'].head(10)

tweets_df = tweets_df.reset_index(drop=True)
tweets_df.head()

from sklearn.feature_extraction. text import TfidfVectorizer, CountVectorizer

tweets_df['Review']

tweets_df.head()

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

sns.set_style('whitegrid')
# %matplotlib inline

# Assuming 'Review' column is a list of words, join them into a string
tweets_df['Review'] = tweets_df['Review'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

stop = stop + ['is', 'was', 'has', 'have', 'to', 'from']

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
count_data = count_vectorizer.fit_transform(tweets_df['Review'])

# Visualize the 20 most common words
plot_20_most_common_words(count_data, count_vectorizer)

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

def get_top_n_bigram(corpus, n=None) :
  vec = CountVectorizer(ngram_range=(2, 4), stop_words=stop).fit(corpus)
  bag_of_words = vec.transform(corpus)
  sum_words = bag_of_words.sum(axis=0)
  words_freq =[(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
  words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
  return words_freq[:n]

common_words = get_top_n_bigram(tweets_df['Review'] , 8)
mydict={}
for word, freq in common_words:
  bigram_df = pd.DataFrame(common_words,columns = ['ngram', 'count'])

bigram_df.groupby( 'ngram' ).sum()['count'].sort_values(ascending=False).sort_values().plot.barh(title = 'Top 8 bigrams',color='orange' , width=.4, figsize=(12,8),stacked = True)