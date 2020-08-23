import numpy as np
import sys
import csv
from joblib import load, dump
from trainers import extract_features_test
import pandas as pd



TWEET_LABEL = 'tweet'
TWEET_IND = 1

def classify(test_set):

    data_set = pd.DataFrame()
    data_set["tweet"] = test_set
    features = extract_features_test(data_set ,"chosenFeatureGetter.pkl")
    classifier = load('chosenClassifier.pkl')
    predictions = classifier.predict(features)
    print(predictions)
    return predictions


def getTweetsAsNpArray(path):
    with open(path,'r') as dest_f:
        data_iter = csv.reader(dest_f)
        data = [data for data in data_iter]
        data_array = np.asarray(data)
    if(data_array[0,0]=='tweet'):
        data_array = np.delete(data_array,0,0)
    return data_array




if __name__ == '__main__':
    tweets = getTweetsAsNpArray(sys.argv[1])
    predictions = classify(tweets[:, 0])
