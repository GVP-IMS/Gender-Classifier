### CORPUS MODULE OVERVIEW ###
# Contain a bunch of methods for preprocessing as well as the main program classes of User and Tweet
# read_file and to_training_data are used initially to prepare an experiment and can later be skipped, read Main for more info

# Import Space

import csv
import re
from Stopwords import * 

# The 2 main classes of the program
class Tweet:
    def __init__(self, handle, content):
        self.handle = handle
        self.content = content

class User:
    def __init__(self, handle, true_gender, assigned_gender):
        self.handle = handle
        self.true_gender = true_gender
        self.assigned_gender = assigned_gender
        self.tweets = []

# 1. Select Training Data Method: Read csv file responding to a chosen number (i) training fold and return training tuples
# fold number 1 = first 10% of data was used for testing -> the training fold contains the last 90%
# Tuples look like: [['handle1', 'unprocessed tweet1'], ['handle2', 'unprocessed tweet2'] ... ]
def read_file(i: int, lan: str):
    # The list of tuples to return
    handle_tweet = []

    # Read training data file
    with open(f'../Data/Partitions/train_data_fold_{i}_{lan}.csv', mode='r', newline='', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            columns = line.split('\t')
            # column 5 = handles; column 8 = tweets
            handle_tweet.append([columns[5], columns[8]])

    # Remove tweets of authors with unknown gender
    true_gender_dict = {}
    with open(f'../Data/Original/{lan.upper()}/{lan}_gender_info.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            cols = line.strip().split('\t')
            true_gender_dict[cols[0]] = cols[1]
    for i in handle_tweet:
        try:
            true_gender_dict[i[0]]
        except KeyError:
            handle_tweet.remove(i)

    return handle_tweet

# 2. Tokenize Method: Used to "recognize" our tokens and combined with the other filters
def tokenizer(text: str, lan: str):
    tokenize_output = []
    text = text.replace('[NEWLINE]', ' ').replace('NEWLINE', ' ')

    # find the following patterns in the string
    emoticon_pattern = re.compile(
        r'(?!\w:)(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)',
        flags=re.IGNORECASE)
    url_pattern = re.compile(r"\b(?:https?://|www\.)\S+\b")
    token_pattern = re.compile(r"\B@\w+|\b\w+\b|")

    for token in emoticon_pattern.findall(text):
        text = text.replace(token, '')
    for token in url_pattern.findall(text):
        text = text.replace(token, '')
    for token in token_pattern.findall(text):
        token = token.lower()
        if token != '':
            tokenize_output.append(token)

    stop_word_list = {
        'de': stop_word_list_de,
        'fr': stop_word_list_fr,
        'it': stop_word_list_it,
        'nl': stop_word_list_nl
    }.get(lan, [])

    # remove replies, mentions and stop words
    for i in tokenize_output:
        if i.startswith('@'):
            tokenize_output.remove(i)
        if i in stop_word_list:
            tokenize_output.remove(i)
    return tokenize_output

# Get training tuples, preprocess them and save them in a file
def to_training_data(i: int, lan: str):
    handle_tweet_list = read_file(i, lan)
    true_gender_dict = {}
    with open(f'../Data/Original/{lan.upper()}/{lan}_gender_info.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            cols = line.strip().split('\t')
            true_gender_dict[cols[0]] = cols[1]

    # create training data list [([processed tweet], 'TRUE GENDER',), (...), ...]
    training_data = []
    with open('../Data/key words_en.txt', 'r', encoding='utf-8') as f:
        english_words = eval(f.read())

    for handle_tweet in handle_tweet_list:
        handle_tweet[1] = tokenizer(handle_tweet[1], lan)
        # English filter
        cnt = 0
        for token in handle_tweet[1]:
            if token in english_words:
                cnt += 1
        if cnt <= 2:
            true_gender = true_gender_dict[handle_tweet[0]]
            training_data.append(tuple([handle_tweet[1], true_gender]))

    # write results into a file
    with open(f'../Data/Training/training data({i})_{lan}.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(['tweet', 'gender'])  # Write the header row only if the file is empty
        for tp in training_data:
            tweet, gender = tp
            writer.writerow([tweet, gender])
