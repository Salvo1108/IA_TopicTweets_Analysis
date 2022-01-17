import json
import re
from sre_parse import Tokenizer

import pandas as pd

#Lista tweet
tweetList = []
print("Started Reading JSON file which contains multiple JSON document")

#Lettura tweets da file JSON
with open("./dataset/2020-10/2020-10-18-22.json") as tweets:
    for jsonObj in tweets:
        tweetDict = json.loads(jsonObj)
        tweetList.append(tweetDict)

#Stampa tweets letti e inseriti nella lista
print("Printing each JSON Decoded Object")
for tweet in tweetList:
    print(tweet["full_text"])

#Create a dataframe with a colum called Tweets

df = pd.DataFrame([tweet["full_text"] for tweet in tweetList], columns=['Tweets'])

#Show the first 5 rows of data
print(df.head())


#Clean the text
#Creaazione di una funzione per la pulizia dei tweets
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  #Removed @mentions
    text = re.sub(r'#', '',text) #Removing the '#' symbol
    text = re.sub(r':', '', text)  # Removing the ':' symbol
    text = re.sub(r'/', '', text)  # Removing the '/' symbol
    text = re.sub(r'\n', '', text)  # Removing the '\' symbol
    text = re.sub(r'RT[\s]+','',text) #Removing RT
    text = re.sub(r'https?:\/\/\S+', '', text) #Remove the hyper link

    return text

df['Tweets']= df['Tweets'].apply(cleanTxt)

print(df)
