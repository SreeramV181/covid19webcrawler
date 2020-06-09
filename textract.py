#Insert code to extract relevant text from websites

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from time import sleep
from sklearn.naive_bayes import GaussianNB
from utils import extract_text_from_urls, get_urls_classes_from_file, bagOfWords, bucketize_classification
from sklearn.metrics import confusion_matrix, classification_report

def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total {} points : {}".format(X_test.shape[0], (y_test != y_pred).sum()))
    print(classification_report(y_test, y_pred))

def main():
    URLS, classifications = get_urls_classes_from_file("Rawcrawl_data.csv")

    # Convert classifications to integer buckets
    bucketized_classifications = np.array(bucketize_classification(classifications[:109]))

    #Extract text from websites
    extracted_text = extract_text_from_urls(URLS[:109])

    #Create bag of words features
    text_counts = bagOfWords(extracted_text).toarray()
    with open('text_counts.pickle', 'wb') as f:
        pickle.dump(text_counts, f)
    with open('classes.pickle', 'wb') as g:
        pickle.dump(classifications, g)

    train_classifier(text_counts, bucketized_classifications)



if __name__ == "__main__":
    main()
