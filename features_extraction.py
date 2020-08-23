## We ended up not using this file. It is submitted only as a testimony to our hard work.



# import textstat
# from textblob import TextBlob
# import pickle
# import numpy as np
# from copy import deepcopy
# from nltk import word_tokenize
import sys
import spacy
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from unigram_LM import *
from create_labls import *
import pandas as pd
import file_to_dataframe
import textstat
from textblob import TextBlob


def get_unigram_mat(tweets, nlp, labels):
    unigram_mat = pd.DataFrame(columns=[str(i) for i in range(10)])
    for i, tweet in enumerate(tweets):
        unigram_mat.loc[i] = get_prob_vector(tweet, nlp, labels)
    return unigram_mat


def extract_features_train(data_frame, small_df=False):

    # first parse
    parsed = parse_data(data_frame=data_frame)

    #generate unigram files
    nlp = spacy.load("en_core_web_sm")
    labels_vec = [i for i in range(10)]
    create_unigram(parsed, nlp, labels_vec, [1 for i in range(10)])

    #generate unigram features
    unigram_mat = get_unigram_mat(parsed.tweet, nlp, labels_vec)

    # generate TF-IDF extractor
    tf_idf_clsf = TfidfVectorizer(stop_words='english',
                           token_pattern=r"\b\w\w+\b|[!|\?|\.|@|#]+|.",
                           binary = True, min_df=3)
    ## generate TF-IDF features
    TF_features = tf_idf_clsf.fit_transform(parsed.tweet)

    dump(tf_idf_clsf, "TF-IDF.joblib")

    # Generate the semantic features
    # sentiment, correct, language = [0] * len(parsed.tweet), [0] * len(parsed.tweet), [0] * len(parsed.tweet)
    # for i, tweet in enumerate(parsed.tweet):
    #     sentiment[i] = TextBlob(tweet).sentiment # Returns the polarity and subjectivity of the text.
    #     correct[i] = TextBlob(tweet).correct() # Returns the text after spelling correction
    #     language[i] = TextBlob(tweet).detect_language() # Returns the detected language of the text
    #     print(sentiment[i], correct[i], language[i])
    #
    # # concat features
    # IDF_features = pd.DataFrame(TF_features, columns = [tf_idf_clsf.get_feature_names()])
    # new = parsed.append(unigram_mat)
    result = pd.concat([parsed, unigram_mat], axis=1, join_axes=[parsed.index])
    # mat = pd.concat([ parsed, unigram_mat], join_axes=)
    # if small_df:
    #     mat.to_csv("small_df_tester_output_chen_and_daniel_champions.csv")
    return result


def extract_features_test(dataframe):

    # first parse
    parsed = parse_data(data_frame=data_frame)

    #generate unigram features
    unigram_mat = get_unigram_mat(parsed.tweet, nlp, labels_vec)

    # # generate TF-IDF features
    # tf_ids = load("TF-IDF.joblib")
    # TF_features = tf_ids.transform(parsed.tweet)

    result = pd.concat([parsed, unigram_mat], axis=1, join_axes=[parsed.index])

    return result


if __name__ == '__main__':

    data_frame = pd.read_csv(sys.argv[1])
    extract_features_train(data_frame, True)
    