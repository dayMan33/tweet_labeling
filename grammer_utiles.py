## We ended up not using this file. It is submitted only as a testimony to our hard work.

import  spacy
from nltk import word_tokenize



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
    sentence = nlp(str(sentence))
    ners = set(" ".join([s.text for s in sentence.ents]).split())
    nouns = (" ").join([token.lemma_ for token in sentence if filter_nouns(token, ners)])
    return nouns


def main():
    nlp = spacy.load("en_core_web_sm")
    twitte = ""
    nouns = get_nouns(twitte, nlp)
    tokenize = list(word_tokenize(nouns))