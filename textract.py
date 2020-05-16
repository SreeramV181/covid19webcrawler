#Insert code to extract relevant text from websites

import pandas as pd
import requests
import pickle
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from time import sleep
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def extract_text_from_urls(URLS):
    webpage_texts = []
    for url in URLS:
        currURL = ""
        try:
            res = requests.get(url)
            html_page = res.content
            soup = BeautifulSoup(html_page, 'html.parser')
            for paragraph in soup.find_all('p'):
                currURL += str(paragraph.text) + " "
            webpage_texts.append(currURL)
        except:
            print("Connection refused")
    return webpage_texts

def bagOfWords(extracted_text):
    #tokenizer to remove unwanted elements from out data like symbols and numbers
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
    text_counts= cv.fit_transform(extracted_text)
    return text_counts

def train_classifier(X, y):
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=.2)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d" (X_test.shape[0], (y_test != y_pred).sum()))

def bucketize_classification(classifications):
    # Correct = 0
    # Error: Related product = 1
    # Error: Unrelated product = 2
    bucketized_classifications = []
    for category in classifications:
        if category == "Correct":
            bucketized_classifications.append(0)
        elif category == "Error: Related product":
            bucketized_classifications.append(1)
        else:
            bucketized_classifications.append(2)
    return bucketized_classifications


def main():
    URLS = []
    classifications = []
    URLS_df = pd.read_csv('Rawcrawl_data.csv', sep=",") #.tolist()
    classifications_df = pd.read_csv('Rawcrawl_data.csv', sep=",") #.tolist()
    URLS = URLS_df['Webpage'].to_list()
    classifications = classifications_df['Class'].to_list()

    # Convert classifications to integer buckets
    bucketized_classifications = bucketize_classification(classifications[:109])

    #Extract text from websites
    extracted_text = extract_text_from_urls(URLS[:109])
    # print(extracted_text)

    #Create bag of words features
    text_counts = bagOfWords(extracted_text)
    with open('text_counts.pickle', 'wb') as f:
        pickle.dump(text_counts, f)
    with open('classes.pickle', 'wb') as g:
        pickle.dump(classifications, g)

    train_classifier(text_counts, classifications)



if __name__ == "__main__":
    main()
