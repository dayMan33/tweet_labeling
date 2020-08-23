## We ended up not using this file. It is submitted only as a testimony to our hard work.


import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
import copy
import matplotlib.pyplot as plt
import os
import sys
import zipfile
import datetime
from sklearn.model_selection import train_test_split
from run_classifier import *


VALIDATION_RATIO = 0.1
RANDOM_STATE = 9527


def main():
    VOCAB = '../input/huggingfacepytorchpretrainedbertchinese/bert-base-chinese-vocab.txt'
    MODEL = '../input/huggingfacepytorchpretrainedbertchinese/bert-base-chinese'

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased') #maybe uncased would yield better results?
    # Tokenized input
    text = "[CLS]" + twitt + "[SEP]"
    tokenized_text = tokenizer.tokenize(text)
    ### TODO: load the data. how?
    train = dataframe
    test = pd.read_csv(TEST_CSV_PATH, index_col='id')

    # divide the train set to train and val - NOT SURE WE NEED
    train, val= train_test_split(train,
            test_size=VALIDATION_RATIO,
            random_state=RANDOM_STATE)

    train_examples = [InputExample('train', twitt, label=dataframe[]) for twitt in train.itertuples()]
    val_examples = [InputExample('val', twitt, label=dataframe[]) for twitt in val.itertuples()]
    test_examples = [InputExample('test', twitt, label=dataframe[]) for twitt in test.itertuples()]

    ## train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    gradient_accumulation_steps = 1
    train_batch_size = 32
    eval_batch_size = 128
    train_batch_size = train_batch_size // gradient_accumulation_steps
    output_dir = 'output'
    bert_model = 'bert-base-cased'
    num_train_epochs = 3
    num_train_optimization_steps = int(
        len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
    cache_dir = "model"
    learning_rate = 5e-5
    warmup_proportion = 0.1
    max_seq_length = 128
    label_list = ['unrelated', 'agreed', 'disagreed']

    tokenizer = BertTokenizer.from_pretrained(VOCAB)
    model = BertForSequenceClassification.from_pretrained(MODEL,
                                                          cache_dir=cache_dir,
                                                          num_labels = 3)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
