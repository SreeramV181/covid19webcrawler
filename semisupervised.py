import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from utils import *
import pickle as pickle

def train_classifier(X, y):
    n_total_samples = len(y)
    n_labeled_points = 109

    indices = np.arange(n_total_samples)

    unlabeled_set = indices[n_labeled_points:]

    # #############################################################################
    # Shuffle everything around
    y_train = np.copy(y)
    y_train[unlabeled_set] = -1

    # #############################################################################
    # Learn with LabelSpreading
    lp_model = LabelSpreading()#gamma=.25, max_iter=20)
    lp_model.fit(X, y_train)
    predicted_labels = lp_model.transduction_[unlabeled_set]
    true_labels = y[unlabeled_set]

    cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)

    print("Label Spreading model: %d labeled & %d unlabeled points (%d total)" %
          (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples))

    print(classification_report(true_labels, predicted_labels))

    print("Confusion matrix")
    print(cm)

    # Plot output labels
    outer, inner = 0, 1
    labels = np.full(n_total_samples, -1.)
    labels[0] = outer
    labels[-1] = inner

    plt.figure(figsize=(8.5, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[labels == outer, 0], X[labels == outer, 1], color='navy',
                marker='s', lw=0, label="outer labeled", s=10)
    plt.scatter(X[labels == inner, 0], X[labels == inner, 1], color='c',
                marker='s', lw=0, label='inner labeled', s=10)
    plt.scatter(X[labels == -1, 0], X[labels == -1, 1], color='darkorange',
                marker='.', label='unlabeled')
    plt.legend(scatterpoints=1, shadow=False, loc='upper right')
    plt.title("Raw data (2 classes=outer and inner)")

    plt.subplot(1, 2, 2)
    predicted_labels_array = np.asarray(predicted_labels)
    outer_numbers = np.where(predicted_labels_array == outer)[0]
    inner_numbers = np.where(predicted_labels_array == inner)[0]
    plt.scatter(X[outer_numbers, 0], X[outer_numbers, 1], color='navy',
                marker='s', lw=0, s=10, label="outer learned")
    plt.scatter(X[inner_numbers, 0], X[inner_numbers, 1], color='c',
                marker='s', lw=0, s=10, label="inner learned")
    plt.legend(scatterpoints=1, shadow=False, loc='upper right')
    plt.title("Labels learned with Label Spreading (KNN)")

    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.92)
    plt.show()

def main():
    URLS, classifications = get_urls_classes_from_file("Rawcrawl_data.csv")

    # Convert classifications to integer buckets
    bucketized_classifications = np.array(bucketize_classification(classifications))

    #Extract text from websites
    extracted_text = extract_text_from_urls(URLS)

    #Create bag of words features
    text_counts = bagOfWords(extracted_text).toarray()
    with open('text_counts.pickle', 'wb') as f:
        pickle.dump(text_counts, f)
    with open('classes.pickle', 'wb') as g:
        pickle.dump(bucketized_classifications, g)

    train_classifier(text_counts, bucketized_classifications)

if __name__ == "__main__":
    main()
