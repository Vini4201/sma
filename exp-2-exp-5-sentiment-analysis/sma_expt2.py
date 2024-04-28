
!pip install pytube

# Import necessary libraries
import os
import json
import googleapiclient.discovery
import pandas as pd
from textblob import TextBlob
from pytube import YouTube

# Specify your YouTube Data API key
DEVELOPER_KEY = "AIzaSyCk5y59_XIlabK3bDipO7FdPSD7lBguxIA"

# Function to get video comments
def get_video_comments(video_id):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="id,snippet",
        maxResults=100,
        order="relevance",
        videoId=video_id
    )
    response = request.execute()

    return response

# User input for the YouTube video link
video_link = input("Enter the YouTube video link: ")

# Extract video ID from the YouTube link
yt = YouTube(video_link)
video_id = yt.video_id

# Get video comments using the extracted video ID
comments = get_video_comments(video_id)

# Print the response as a JSON string
print(json.dumps(comments, indent=4))

import pandas as pd

def create_df_author_comments(response):
    authorname = []
    comments = []
    for i in range(len(response["items"])):
        authorname.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"])
        comments.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
    df_1 = pd.DataFrame(comments, index=authorname, columns=["Comments"])
    return df_1

response = get_video_comments(video_id)
df_1 = create_df_author_comments(response)
df_1

from textblob import TextBlob

def perform_sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis to the 'Comments' column in df_1
df_1["Sentiment"] = df_1["Comments"].apply(perform_sentiment_analysis)

# Print the resulting DataFrame with sentiment analysis
print(df_1)

from textblob import TextBlob

def calculate_sentiment_score(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Apply sentiment analysis to the 'Comments' column in df_1 and calculate sentiment scores
df_1["SentimentScore"] = df_1["Comments"].apply(calculate_sentiment_score)

# Calculate the overall sentiment score of the video
overall_sentiment_score = df_1["SentimentScore"].mean()

# Determine the overall sentiment label based on the sentiment score
if overall_sentiment_score > 0:
    overall_sentiment_label = "Positive"
elif overall_sentiment_score < 0:
    overall_sentiment_label = "Negative"
else:
    overall_sentiment_label = "Neutral"

# Print the overall sentiment of the video
print("Overall Sentiment: {} (Score: {})".format(overall_sentiment_label, overall_sentiment_score))

