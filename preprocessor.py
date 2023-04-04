import re
import csv
import pandas as pd
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk


# Cleans up the text by excluding special characters and stopwords.
def word_extraction(sentence):
    ignore = ["subject"]
    words = re.sub("[^\w]", " ", sentence).split()
    # cleaned_text = [w.lower() for w in words if w.lower() not in ignore]
    cleaned_text = [w.lower() for w in words if w.lower() not in stopwords.words('english')]
    # print("cleaned_text: ", cleaned_text)
    return cleaned_text


# Vectorize each line of the dataset in a feature-set of length 768 by utilizing LaBSE from Sentence Transformers.
def vectorize_text(filepath):
    model = SentenceTransformer('sentence-transformers/LaBSE')
    # model = SentenceTransformer('all-MiniLM-L6-v2')

    f = open(filepath, "r")
    fw = open("vectorized_dataset.csv", "w")
    reader = csv.reader(f)
    writer = csv.writer(fw)

    headline = []
    for i in range(768):
        headline.append("s" + str(i))

    headline.append('label')
    writer.writerow(headline)

    count = 0
    count1 = 0
    for line in reader:
        count1 += 1
        if count1 == 1:
            print(line)
            continue
        # print(line[2])
        word_list = word_extraction(line[2])
        # print(word_list)
        text = ' '.join(word_list)
        # print(text)
        if "subject" in text:
            count += 1
        vector = model.encode(text)
        row = []
        row.extend(vector)
        row.extend(line[3])
        writer.writerow(row)

    print("Total mails: ", count1)
    print("Mails that contain the word 'subject': ", count)

    f.close()
    fw.close()
    return


def print_prior_stats(X, y):
    samples_n = len(y)
    ham_n = 0
    spam_n = 0
    count = 0
    for sample in y:
        if sample == 0:
            ham_n += 1
        elif sample == 1:
            spam_n += 1
        else:
            print("???????????, ,,,,,, ", sample)
            sys.exit("Error, label sample is neither 0 nor 1. Exiting!")
        count += 1
    if count != samples_n:
        sys.exit("Error, miscount of labels. Exiting!")
    print("Total samples: ", samples_n)
    print("Spam: ", spam_n, "(", (spam_n / samples_n) * 100, "%)")
    print("Ham : ", ham_n, "(", (ham_n / samples_n) * 100, "%)")
    return


def open_dataset(dataset_path):
    f = open("vectorized_dataset.csv", "r")
    reader = csv.reader(f)

    skip_flag = True

    df = pd.read_csv(dataset_path)

    X = df.drop(["label"], axis=1).to_numpy()
    y = df["label"].to_numpy()
    # print(X)
    # print(y)

    return X, y


