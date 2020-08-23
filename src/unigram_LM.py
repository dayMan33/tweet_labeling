## We ended up not using this file. It is submitted only as a testimony to our hard work.


import spacy
from tqdm import tqdm
import pickle
from collections import defaultdict
import numpy as np
from copy import deepcopy
import pandas as pd
from nltk import word_tokenize


PATH_TO_PREPROCEED_FILE = r"/cs/labs/dshahaf/chenxshani/IML/IML_Hackathon_2019/rrrr.csv"


def filter_nouns(token, ners):
    """
    Filters the nouns
    """
    if token.pos_ != "NOUN":
        return False
    if not token.lemma_.isalpha():
        return False
    if token.text in ners:
        return False
    if token.text == token.text.upper():
        return False
    return True


def get_nouns(sentence, nlp):
    """
    Gets the nouns
    """
    sentence = nlp(str(sentence))
    ners = set(" ".join([s.text for s in sentence.ents]).split())
    nouns = (" ").join([token.lemma_ for token in sentence if filter_nouns(token, ners)])
    return nouns


def unigram_stats(dataframe, nlp, label):
    """
    Computes the uni-gram model for the datafile
    """
    nouns_dict = defaultdict(int)
    for twitt in tqdm(dataframe["tweet"]):
        doc = nlp(str(twitt.lower()))
        nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN" if token.lemma_.isalpha()]
        for noun in nouns:
            nouns_dict[noun] += 1
    return nouns_dict


def dict_of_dicts_merge(x, y):
    z = x.copy()
    overlapping_keys = x.keys() & y.keys()
    for key in overlapping_keys:
        z[key] = x[key] + y[key]
    for key in y.keys() - overlapping_keys:
        z[key] = deepcopy(y[key])
    return z


def normalization_unigram(label, p_label):
    """
    Normalizes the uni-grams model and return a logit distribution for each sequence
    """
    with open("unigram_stats_{}.pickle".format(label), 'rb') as f:
        unigram_dict = pickle.load(f)
        values = unigram_dict.values()
        counter = sum(values)
    with open("all_labels_final.pickle", 'rb') as file:
        all_labels = pickle.load(file)
        for key in unigram_dict.keys():
            unigram_dict[key] = np.log(unigram_dict[key]) - np.log(counter) - np.log(all_labels[key]) #+ np.log(p_label)

    with open("unigram_stats_normalized_{}.pickle".format(label),"wb") as pickle_out:
        pickle.dump(unigram_dict, pickle_out)


def dict_of_dicts_merge(x, y):
    z = x.copy()
    overlapping_keys = x.keys() & y.keys()
    for key in overlapping_keys:
        z[key] = x[key] + y[key]
    for key in y.keys() - overlapping_keys:
        z[key] = deepcopy(y[key])
    return z


def inter_fields(labels):
    with open("unigram_stats_0.pickle", 'rb') as f:
        label_0 = pickle.load(f)
        with open("unigram_stats_1.pickle", 'rb') as l:
            label_1 = pickle.load(l)
            tmp = dict_of_dicts_merge(label_0, label_1)
    for label in labels[2:]:
        with open("unigram_stats_{}.pickle".format(label), 'rb') as f:
            dict = pickle.load(f)
            tmp = dict_of_dicts_merge(dict, tmp)
    with open("all_labels_final.pickle", "wb") as pickle_out:
        pickle.dump(tmp, pickle_out)


def pickle_creation(nlp, labels_vector, p_labels, datafile):
    for label in labels_vector:
        with open("unigram_stats_{}.pickle".format(label), 'wb') as f:
            cond = (datafile.user == label)
            df = datafile[cond]
            dict_label = unigram_stats(dataframe=df, nlp=nlp, label=label)
            pickle.dump(dict_label, f)


def get_prob_vector(sentence, nlp, labels_vector):
    nouns = get_nouns(sentence, nlp)
    print(nouns)
    tokenize = list(word_tokenize(nouns))
    probability_matrix = np.zeros((len(labels_vector), len(tokenize)))
    for label in labels_vector:
        with open("unigram_stats_normalized_{}.pickle".format(label), "rb") as f:
            dict = pickle.load(f)
            for i, token in enumerate(tokenize):
                probability_matrix[label, i] = np.exp(dict[token])
    mean_prob_matrix = np.mean(probability_matrix, axis=1)
    print(probability_matrix)
    return mean_prob_matrix


def main():

    # Creates a pickle file of the distribution of nouns for each label in labels_vector and for all together
    nlp = spacy.load("en_core_web_sm")
    # p_labels = [0.2921658986175, 0.1105990783410, 0.0359447004608, 0.2211981566820, 0.1705069124424,
    #             0.0341013824885, 0.0101382488479, 0.0138248847926, 0.0940092165899, 0.0175115207373] # Original!
    # p_labels = [0.2021658986175, 0.1905990783410, 0.0309447004608, 0.2911981566820, 0.2305069124424,
    #             0.0391013824885, 0.0191382488479, 0.0198248847926, 0.100092165899, 0.0235115207373]
    # datafile = pd.read_csv(PATH_TO_PREPROCEED_FILE)
    labels_vector = [0,1,2,3,4,5,6,7,8,9]

    # pickle_creation(nlp, labels_vector, p_labels, datafile)

    # Normalize the dicts to logit distribution
    # inter_fields(labels_vector)
    # for i in range(len(labels_vector)):
    #     normalization_unigram(labels_vector[i], p_labels[i])

    # Create word embedding for each word in a tweet:
    # tweet = "Just added one! @shtat: Arnold @Schwarzenegger is taking requests to recite movie lines and sharing the results:"
    # tweet = "KimKardashian i usually hated sundays, but with  i love them"
    # tweet = "GBack from Japan after a very successful trip. Big progress on MANY fronts. A great country with a wonderful leader in Prime Minister Abe!"
    # tweet = "Light peach, pinky peach, mid-tone coral, burnt red. Which Peach Cr me Lipstick are you excited for? Coming to the Pop-Up TO "
    mean_prob_matrix = get_prob_vector(sentence=tweet, nlp=nlp, labels_vector=labels_vector)
    # print(mean_prob_matrix, sum(mean_prob_matrix))git add-A


if __name__ == "__main__":
    main()

