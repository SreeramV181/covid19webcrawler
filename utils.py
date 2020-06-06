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
    URLS_df = pd.read_csv(filename, sep=",")
    classifications_df = pd.read_csv(filename, sep=",")
    URLS = URLS_df['Webpage'].to_list()
    classifications = classifications_df['Class'].to_list()
    return (URLS, classifications)

def bagOfWords(extracted_text):
    #tokenizer to remove unwanted elements from out data like symbols and numbers
    token = RegexpTokenizer(r'[a-zA-Z]+')

    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range = (1,1), min_df = 1, tokenizer=token.tokenize)
    text_counts = vectorizer.fit_transform(extracted_text)
    # cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range = (1,1), min_df=3, tokenizer = token.tokenize)
    # text_counts= cv.fit_transform(extracted_text)
    return text_counts

def bucketize_classification(classifications):
    bucketized_classifications = []
    for category in classifications:
        if category == "Correct" or category == "Error: Related product":
            bucketized_classifications.append(0)
        else:
            bucketized_classifications.append(1)
    return bucketized_classifications
