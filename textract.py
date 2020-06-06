#Insert code to extract relevant text from websites

import pandas as pd
import requests
import pickle
import numpy as np
from bs4 import BeautifulSoup
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from time import sleep
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer

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
            webpage_texts.append("") # to make sure shape is consistent
            print("Connection refused")
    return webpage_texts

def get_urls_classes_from_file(filename):
    URLS = []
    classifications = []
    URLS_df = pd.read_csv(filename, sep=",") #.tolist()
    classifications_df = pd.read_csv(filename', sep=",") #.tolist()
    URLS = URLS_df['Webpage'].to_list()
    classifications = classifications_df['Class'].to_list()
    return (URLS, classifications)


def bagOfWords(extracted_text):
    #tokenizer to remove unwanted elements from out data like symbols and numbers
    token = RegexpTokenizer(r'[a-zA-Z]+')

    # ## Compare TFIDF vs CountVectorizer
    # # TFIDF
    # vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range = (1,1), min_df = 2, tokenizer=token.tokenize)
	# text_counts = vectorizer.fit_transform(extracted_text)
	# print("TFIDF Vocab")
	# print(vectorizer.vocabulary_)

    # CountVectorizer
    cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range = (1,1), min_df=2, tokenizer = token.tokenize)
    text_counts= cv.fit_transform(extracted_text)
    # print(cv.get_feature_names())
    # #print(cv.vocabulary_)
    return text_counts

def train_classifier(X, y):
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total {} points : {}".format(X_test.shape[0], (y_test != y_pred).sum()))

def bucketize_classification(classifications):
    # Correct = 0
    # Error: Related product = 1
    # Error: Unrelated product = 2
    bucketized_classifications = []
    for category in classifications:
        if category == "Correct" or category == "Error: Related product":
            bucketized_classifications.append(0)
        # elif category == "Error: Related product":
        #     bucketized_classifications.append(1)
        else:
            bucketized_classifications.append(1)
    return bucketized_classifications


def main():
    URLS, classifications = get_urls_classes_from_file("Rawcrawl_data.csv")

    # Convert classifications to integer buckets
    bucketized_classifications = np.array(bucketize_classification(classifications[:109]))

    #Extract text from websites
    extracted_text = extract_text_from_urls(URLS[:109])
    # print(extracted_text)

    #Create bag of words features
    text_counts = bagOfWords(extracted_text).toarray()
    with open('text_counts.pickle', 'wb') as f:
        pickle.dump(text_counts, f)
    with open('classes.pickle', 'wb') as g:
        pickle.dump(classifications, g)

    train_classifier(text_counts, bucketized_classifications)



if __name__ == "__main__":
    main()
