## We ended up not using this file. It is submitted only as a testimony to our hard work.

import re

import pandas as pn
# from emoji.unicode_codes import UNICODE_EMOJI
cur_path = r"C:\Users\Nitzan\Documents\TOV\trainData"
emoji_regex = re.compile(r"["
                         u"\U0001F600-\U0001F64F"  # emoticons
                         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                         u"\U0001F680-\U0001F6FF"  # transport & map symbols
                         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                         u"\U00002702-\U000027B0"
                         u"\U000024C2-\U0001F251"
                         "]", flags=re.UNICODE)

with_shortcut_regex = re.compile(r'(?:^|[\s]+)w/(?:^|[\s]+)')
dollar_shortcut_regex = re.compile(r'[\d]+\$(?:^|[\s]+)')
re_tweet_regex = re.compile(r"^RT[\s]*@")

url_regex = re.compile(r"((?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))("
                       r"?:["
                       r"\w.,@?^=%&:/+#-]*[\w@?^=%&/+#-])?)")
hash_tag_regex = re.compile(r"(?:^|\W|\D|\s])(#[\S]+)\b")
tag_regex = re.compile(r"(?:^|\W|\D|\s])(@[\S]+)\b")
punctuation_regex = re.compile(r"(\!)")
start_with_capital_letter_regex = re.compile(r"^[\s]*[A-Z]")
whole_word_capital_regex = re.compile(r"(?:[\s]+|^)[A-Z]{2,}(?:$|[\s]+)")


def retweet_capital_letters_lable_maker(data_frame):
    '''make a capital letter(start and wholeword)lable , retweet labls and
    clean the retweet  ****and panctuation mark'''
    RT_indicator_index = []
    start_with_capital_letter = []
    whole_word_capital_num = []
    retweet_tag = []
    new_tweets = []
    num_of_chars = []
    punctuation_arr = []
    for cur_tweet in data_frame.tweet:
        if (re_tweet_regex.match(cur_tweet)):
            RT_indicator_index.append(1)
            retweet_tag.append(
                cur_tweet[cur_tweet.find("@") + 1:cur_tweet.find(
                    ":")])
            cur_tweet = cur_tweet[cur_tweet.find(":") + 1:]
        else:
            RT_indicator_index.append(0)
            retweet_tag.append(None)
        num_of_chars.append(len(cur_tweet))
        cur_tweet = shortcuts_rplacer(cur_tweet)
        new_tweets.append(cur_tweet)
        start_with_capital_letter.append(
            1 if (start_with_capital_letter_regex.match(
                cur_tweet)) else 0)
        whole_word_capital_arr = whole_word_capital_regex.findall(cur_tweet)
        whole_word_capital_num.append(len(whole_word_capital_arr))
        punctuation_arr.append(len(punctuation_regex.findall(cur_tweet)))
    data_frame["tweet"] = new_tweets
    data_frame["num_of_chars"] = num_of_chars
    data_frame["RT_indicator"] = RT_indicator_index
    data_frame["RT_tag"] = retweet_tag
    data_frame["num_of_capital_words"] = whole_word_capital_num
    data_frame["start_with_capital"] = start_with_capital_letter
    data_frame["punctuation_num"] = punctuation_arr
    return data_frame


def shortcuts_rplacer(tweet):
    cur_tweet = with_shortcut_regex.sub(' with ', tweet)  # \w to with
    cur_tweet = dollar_shortcut_regex.sub(' dollars ', cur_tweet)
    return cur_tweet


def create_hashtags_url_tags_labls(data_frame):
    '''create url tags and hash tags labls
    \B mast \B be after  retweet_capital_letters_lable_maker function and
    not before!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
    url_tag = []
    url_num = []
    hash_tags_num = []
    hash_tags = []
    tags_num = []
    tags = []
    tweet_arr = []
    for tweet in data_frame.tweet:
        temp_tweet = tweet
        url_arr = url_regex.findall(tweet)
        for i in range(len(url_arr)):
            temp_tweet = (url_regex.sub(' ', temp_tweet))
        url_num.append(len(url_arr))
        url_tag.append(url_arr)
        hash_tag_arr = hash_tag_regex.findall(tweet)
        for i in range(len(hash_tag_arr)):
            temp_tweet = hash_tag_regex.sub(' ', temp_tweet)
        hash_tags_num.append(len(hash_tag_arr))
        hash_tags.append(hash_tag_arr)
        tags_arr = tag_regex.findall(tweet)
        # for i in range(len(tags_arr)):
        #     temp_tweet = tag_regex.sub(' ',temp_tweet)
        tags_num.append(len(tags_arr))
        tags.append(tags_arr)
        tweet_arr.append(temp_tweet)

    data_frame["tweet"] = tweet_arr
    data_frame["tags_num"] = tags_num
    data_frame["tags"] = tags
    data_frame["hashtags_num"] = hash_tags_num
    data_frame["hashtags"] = hash_tags
    data_frame["url_num"] = url_num
    data_frame["urls"] = url_tag
    return data_frame


def create_emoji_labls(data_frame):
    emoji_arr = []
    emoji_num = []
    tweet_arr = []
    num_of_words = []
    for tweet in data_frame.tweet:
        cur_emoji = emoji_regex.findall(tweet)
        emoji_num.append(len(cur_emoji))
        new_tweet = emoji_regex.sub(' ', tweet)
        new_tweet = re.sub(r'‚Ä¶', '', new_tweet)
        # cur_emoji_discription = []
        # for j in cur_emoji:
        #     try:
        #         j_emoji = UNICODE_EMOJI[j]
        #         j_emoji = j_emoji.replace(":", "")
        #         cur_emoji_discription.append(j_emoji)
        #     except:
        #         continue
        emoji_arr.append(cur_emoji)
        new_tweet = re.sub(r'[^\x00-\x7F]+', ' ', new_tweet)
        tweet_arr.append(new_tweet)
        num_of_words.append(len(tweet.split()))
    data_frame["emoji_num"] = emoji_num
    data_frame["emojis"] = emoji_arr
    data_frame["tweet"] = tweet_arr
    data_frame["num_of_words"] = num_of_words
    return data_frame


def parse_data(data_frame=None, path=None):
    if (path):
        data_frame = file_to_dataframe.first_read(path)
    data_frame = retweet_capital_letters_lable_maker(data_frame)
    data_frame = create_hashtags_url_tags_labls(data_frame)
    data_frame = create_emoji_labls(data_frame)
    return data_frame


# if __name__ == '__main__':
#     data_frame = file_to_dataframe.first_read(cur_path)
#     #     data_frame =dt.sample(frac=0.1)
#     data_frame = parse_data(data_frame=data_frame)
#     data_frame.to_csv(r"C:\Users\Nitzan\Documents\TOV"
#                       r"\IML_Hackathon_2019\rrrr.csv")
