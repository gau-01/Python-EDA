print("Python Project EDA")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
print("Importing data")
feedback=pd.read_csv("customer_feedback_data.csv")
Meta=pd.read_csv("customer_metadata_data.csv")
Info=pd.read_csv("product_information_data.csv")
print("Pre-processing data")
print(feedback.head())
print(feedback.info())
print(feedback.describe())
print(feedback.duplicated())

Data1=Meta.merge(feedback, on="Customer_ID")
Data2=Info.merge(feedback,on="Product_ID")

df=Data2.filter(items=["Product_Name","Feedback_Type","Sentiment_Label","Price"])
print(df)
sentiment=df["Sentiment_Label"]
percentage=100
feedback_type_sentiment = df['Sentiment_Label'].value_counts(normalize=True)*100
product_sentiment = df.groupby('Product_Name')['Sentiment_Label'].value_counts(normalize=True) * 100

#Sentiment Distribution by Feedback Type
df['Sentiment_Label'].value_counts(normalize=True)
sns.barplot(x=sentiment, y=percentage, palette=['blue','red','green'])
plt.title('Sentiment Distribution by Feedback Type')
plt.xlabel('Sentiment_Label')
plt.ylabel('Percentage')
plt.show()


#Sentiment Distribution by Product
product_sentiment.unstack().plot(kind='bar')
plt.title('Sentiment Distribution by Product')
plt.xlabel('Product_Name')
plt.ylabel('Percentage')
plt.figure(figsize=(20,10),dpi=400)
plt.show()

#Word Cloud for Positive Sentiment
sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
feedback['Sentiment_Label'] = feedback['Sentiment_Label'].map(sentiment_map)

from wordcloud import WordCloud

positive_text = feedback[feedback['Sentiment_Label'] > 0]['Comment']
wordcloud = WordCloud(width=800, height=400).generate(' '.join(positive_text))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#Word Cloud for Negative Sentiment

negative_text = feedback[feedback['Sentiment_Label'] < 0]['Comment']
wordcloud = WordCloud(width=800, height=400).generate(' '.join(negative_text))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#Word Cloud for Neutral Sentiment

Neutral_text = feedback[feedback['Sentiment_Label'] == 0]['Comment']
wordcloud = WordCloud(width=800, height=400).generate(' '.join(Neutral_text))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

df1=Data1.filter(items=["Age","Gender","Location","Sentiment_Label"])
print(df1)

age_sentiment=df1.groupby(["Age","Sentiment_Label"]).size()
gender_sentiment=df1.groupby(["Gender","Sentiment_Label"]).size()
location_sentiment=df1.groupby(["Location","Sentiment_Label"]).size()

print("AGE")
age_sentiment.unstack().plot(kind="bar")
plt.title("Sentiment by Age")
plt.show()

gender_sentiment.unstack().plot(kind="bar")
plt.title("Sentiment by Gender")
plt.show()

location_sentiment.unstack().plot(kind="line")
plt.title("Sentiment by Location")
plt.show()

print("End")
