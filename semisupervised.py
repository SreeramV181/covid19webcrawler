import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelSpreading, LabelPropogation







def main():
    model_type = "SPREAD"
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
