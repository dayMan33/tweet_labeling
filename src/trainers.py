# this file is part of Team XXX on IML Hackaton HUJI 2019
# auther: Ofir Shifman

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import accuracy_score
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split



# Classifier imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
from operator import itemgetter
import pandas as pd
from joblib import dump, load




def sample_by_dist(df,sample_size, replace):
    dist1 = [0.2921658986175, 0.1105990783410, 0.0359447004608, 0.2211981566820,0.1705069124424,
            0.0341013824885, 0.0101382488479, 0.0138248847926, 0.0940092165899,0.0175115207373]
    output = pd.DataFrame()
    for label in range(10):
        cond = (df.user == label)
        cur = df[cond]
        num_of_rows = math.ceil(sample_size*dist1[label])
        if replace:
            new_df = pd.DataFrame()
            while (new_df.shape[0] + cur.shape[0]) < num_of_rows:
                new_df = new_df.append(cur, ignore_index=True)
            num_of_rows = num_of_rows - new_df.shape[0]
            new_df = new_df.append(cur.sample(n=num_of_rows, replace=replace),
                                   ignore_index=True)
            output = output.append(new_df)
        else:
            cur = cur.sample(n=num_of_rows,replace = replace)
            output = output.append(cur)
    return output



def train_model_calc_score(X_train, X_test,y_train,y_test, clf):
    clf.fit(X_train.toarray(), y_train)
    y_predict = clf.predict(X_test.toarray())
    return accuracy_score(y_test, y_predict)

def split_Xy(df):
    y = df['user']
    X = df.drop(['user'], axis=1)
    return X, y




def choose_clf(X_train, X_test, y_train, y_test):
    """
    This is the heart of the code. running all classifiers on the training data and comparing them to eventually choose
    the classifier with the best result.
    """
    Ks = [3,21, 101]  # knn options
    trees_nums = [10,120]  # for random forest options
    # depths = [None, 5, 50, 500, 5000]  # for random forest options
    forest_min_sample_size = [25, 500]  # for random forest options

    # Initialize our classifiers
    clfs = {}
    acc = {}
    clfs["gnb"] = GaussianNB()
    clfs["MNB"] = MultinomialNB()
    clfs["BNB"] = BernoulliNB()
    clfs["LR"] = LogisticRegression()
    clfs["SDG"] = SGDClassifier()
    if X_train.shape[0] < 1000000:
        clfs["SVC"] = SVC()
    clfs["LSVC"] = LinearSVC()
    clfs["NSVC"] = NuSVC(kernel='rbf', nu=0.01)
    for i in Ks: #knns
        clfs["KNN" + str(i)] = KNeighborsClassifier(n_neighbors=i)


    for tn in trees_nums:
            # for d in depths:
            for s in forest_min_sample_size:
                clfs["R_F" + str(tn) +','+ str(s) ] = RandomForestClassifier(n_estimators=tn, min_samples_split=s)


    # keep track of progress while running
    for key,val in tqdm(clfs.items()):
        acc[key] = train_model_calc_score(X_train, X_test,y_train, y_test, val)
        # print(key + " and accuracy is: " + str(acc[key]))

    ordered = OrderedDict(sorted(acc.items(), key = itemgetter(1), reverse = True))
    print(ordered)
    res = (ordered.popitem(last = False))[0]
    return clfs[res]






def extract_features_train(df, trial, train_size):
    tf_idf_clsf = TfidfVectorizer(stop_words='english',
                                  token_pattern=r"\b\w\w+\b|[!|\?|\.|@|#]+|.",
                                  binary = True, min_df=3)
    # generate TF-IDF features
    TF_features = tf_idf_clsf.fit_transform(df.tweet)
    TF_labels = df.user
    dump(tf_idf_clsf, "FeatureGetters/featureGetter" + str(trial) + ':sample_size' + str(train_size) + '.pkl')

    return TF_features, TF_labels




def extract_features_test(df, path_to_getter):
    getter = load(path_to_getter)
    features = getter.transform(df.tweet)
    return features




def main():
    train_sizes = [15000, 25000]
    path = r"/cs/labs/dshahaf/chenxshani/IML/DATA/data.csv"
    df = pd.read_csv(path)


    for j in range(5):
        test_df = sample_by_dist(df, 1000, False)  # size of test_df, no replace
        test_base = df.drop(test_df.index) #don't mix train and test_df
        for i in range(len(train_sizes)):
            print(j, train_sizes[i])
            train_df = sample_by_dist(test_base, train_sizes[i],True)  # size of test_df. todo: compare naive replace=true with first take without replace and then add with
            # todo: multiple samples of train per same size
            X_train, y_train = extract_features_train(train_df, j, train_sizes[i])
            X_test, y_test = split_Xy(test_df)
            X_test = extract_features_test(X_test,"FeatureGetters/featureGetter" + str(j) + ':sample_size' + str(train_sizes[i]) + '.pkl')
            chosen = choose_clf(X_train,X_test,y_train,y_test)
            dump(chosen, 'Classifiers/chosen_classifier' + str(j) + ':sample_size' + str(train_sizes[i]) + '.pkl')






if __name__ == "__main__":
    main()
