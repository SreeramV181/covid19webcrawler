#Insert code to extract relevant text from websites

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from time import sleep

# def tag_visible(element):
#     if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
#         return False
#     if isinstance(element, Comment):
#         return False
#     return True
#
# def text_from_html(body):
#     soup = BeautifulSoup(body, 'html.parser')
#     texts = soup.findAll(text=True)
#     visible_texts = filter(tag_visible, texts)
#     return u" ".join(t.strip() for t in visible_texts)
# #
# html = urllib.request.urlopen('http://www.nytimes.com/2009/12/21/us/21storm.html').read()
# print(text_from_html(html))
# https://www.wellcare.com/Kentucky/Providers/Clinical-Guidelines/CCGs/CCG-List

def extract_text_from_urls(URLS):
    webpage_texts = []
    for url in URLS:
        currURL = ""
        try:
            res = requests.get(url)
            html_page = res.content
            soup = BeautifulSoup(html_page, 'html.parser')
            if soup.contains_replacement_characters == True:
                break

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
    train_x, train_y, test_x, test_y = train_test_split(X, y, test_size=.2)


def main():
    URLS = []
    classifications = []
    URLS_df = pd.read_csv('Rawcrawl_data.csv', sep=",") #.tolist()
    classifications_df = pd.read_csv('Rawcrawl_data.csv', sep=",") #.tolist()
    URLS = URLS_df['Webpage'].to_list()
    classifications = classifications_df['Class'].to_list()

    #Extract text from websites
    extracted_text = extract_text_from_urls(URLS[:109])
    print(extracted_text)

    #Create bag of words features
    text_counts = bagOfWords(extracted_text)
    


if __name__ == "__main__":
    main()
