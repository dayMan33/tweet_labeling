# Tweet labeling - 
ML course hackathon project
---------------------------

As part of a hackathon in an Introduction to machine learning course, we were given a dataset of tweets produces by 10 different celebrities (Cristiano Ronaldo, Lady GaGa, Jimmy Kimmel, Ellen Degeneres, Arnold Schwarzenegger, Kim Kardashian, Donald Trump, Conan O'brien, Lebron Jamesn, Joe Biden). 

Our goal was to build a learner that given a new tweet could predict which celebrity produced it. 

The full implementation and results appear in the project.pdf file 



###File list:

1. project.pdf - A detailed explanation of the process we went through while trying to learn task 1 (who tweeted what).
2. src:
    a. requirements.txt - A list of all packages needed to run our code.
    b. classifier.py - The code using the trained classifier in order to classify new tweets.
    c. trainers.py - The main code we used to find the classifier best fit for this task.
    d. feature_extraction.pt - Functions extracting the relevant features from the tweets.
    e. chosenFeatureGetter.pkl - The object in charge of extracting the features out of the tweets in the test set.
    f. chosenClassifier.pkl - The trained classifier for this task.
    g. unigram_LM.py - Creates the pickle files containing distribution of words by user.
    i. unigram_stats_normalized_0-9.pickle - Probability distribution of different words in tweets for each user
    j. grammer_utils.py - Using built-in libraries to implement grammar-related features.
    k. file_to_dataframe.py - Turns all supplied csv files into a single pandas dataframe.
    l. file_data_sperator.py - Code that ran once and separated the data into a training set and a validation set.
    m. create_labls.py - Code cleaning the original tweet from unnecessary characters and extracting basic features.
    n. BERT_LM.py - Sentence embedding using bidirectional NN based on transformed architecture.
    o. unigram_stats_all_labels_0/-1.py - Temporary files used to create the Unigram pickle files.
