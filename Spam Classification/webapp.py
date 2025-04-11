import string
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#NLP Model
df = pd.read_csv(r"C:\Users\Favour\Desktop\Spam Classification\spam.csv", encoding='latin1')
df = df.drop(["Unnamed: 2", "Unnamed: 3","Unnamed: 4"], axis=1)
df.rename(columns= {"v1":"labels", "v2":"message"}, inplace=True)
df.drop_duplicates(inplace=True)
df["labels"] = df['labels'].map({'ham':0, 'spam':1})
print(df.head())

import string
import nltk
from nltk.corpus import stopwords  # Import stopwords at the top
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Download the stopwords if not already downloaded
nltk.download('stopwords')

def clean_data(message):
    # Remove punctuation
    message_without_punc = [character for character in message if character not in string.punctuation]
    message_without_punc = ''.join(message_without_punc)
    
    # Remove stopwords
    stop_words = stopwords.words("english")  # Get English stopwords
    separator = ' '
    return separator.join([word for word in message_without_punc.split() if word.lower() not in stop_words])

# Load your data
df = pd.read_csv(r"C:\Users\Favour\Desktop\Spam Classification\spam.csv", encoding='latin1')
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
df.rename(columns={"v1": "labels", "v2": "message"}, inplace=True)
df.drop_duplicates(inplace=True)
df["labels"] = df["labels"].map({'ham': 0, 'spam': 1})  # Map labels
print(df.head())

# Apply cleaning
df["message"] = df["message"].apply(clean_data)

x = df["message"]
y = df["labels"]

cv = CountVectorizer()
x = cv.fit_transform(x)

print(x)

df = pd.read_csv(r"C:\Users\Favour\Desktop\Spam Classification\spam.csv", encoding='latin1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1": "labels", "v2": "message"})
df.drop_duplicates(inplace=True)

df["labels"] = df["labels"].map({'ham': 0, 'spam': 1})
print(df.head())

# Apply cleaning
df["message"] = df["message"].apply(clean_data)

x = df["message"]
y = df["labels"]

cv = CountVectorizer()
x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

model = MultinomialNB().fit(x_train, y_train)

predictions = model.predict(x_test)


def predict(text):
    labels = ['Not Spam', 'Spam']
    x = cv.transform(text).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v = int(''.join(s))
    return str("This message is looking to be: "+labels[v])

st.title('Spam Classifier')
st.image("26- O que os spammers querem, como o conseguem e como os impedir_0.png")
user_input = st.text_input('Write your message')
submit = st.button('Predict')
if submit:
    answer = predict([user_input])
    st.text(answer)