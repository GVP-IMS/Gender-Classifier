### MAIN MODULE OVERVIEW ###
# Run this to use the program. Contains blocks of code performing individual steps using all the other modules.
# After inputting several options, you train a classifier and use it to classify a list of users, given appropriate training/testing data in the project folders.

#№0# Import space

from Corpus import *
from Evaluation import Evaluator
from NB import *

# Newer Imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, svm

#№1# Choose program options, see documentation for more info

# Language selection using filename suffixes (both for training/testing)
print('1a. Choose a language (de=German / fr=French / it=Italian / nl=Dutch):')
lang = input().lower()
# Fold number (1-10): determines which part of the data is used for testing and which for training e.g. 1 = first 10% is for testing
print('1b. Enter number of testing fold (rest=training):')
fold = int(input())
# Classifier choice: 1 for NB, 2 for SVM
print('1c. Choose classifier (1=NB / 2=SVM):')
option = int(input())
# Concatenation option: 1 for Yes, 2 for No
print('1d. Concatenate test tweets? (1=Yes / 2=No):')
concat = int(input())

#№2# Read tweets-[language].csv and save the text & gender of tweets designated for training (with labelled users) in a separate csv file

# Decline this option to save time when repeating a previous experiment with the same data split
print('2a. Create and save new training csv? (Y/N)')
answer = input()

print('\n> Model started, this may take a while...')
if answer == 'Y':
    # Take parameters from №1
    to_training_data(fold, lang)
    print('2b. Read and saved training data.')
else:
    print('2b. Using old saved data.')

#№3# Read saved training data from №2 and create training objects inside Main

if option == 1:
    training_data = []
    with open(f'../Data/Training/training data({fold})_{lang}.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header row
        next(reader)
        for i, row in enumerate(reader):
            training_content = eval(row[0])
            training_gender = row[1]
            training_instance = (training_content, training_gender)
            training_data.append(training_instance)
    print(f"3. Read training data file and created [{len(training_data)}] training tuples to be used.")
elif option == 2:
    # SVM1
    Corpus = pd.read_csv(f"../Data/Training/training data({fold})_{lang}.csv")
    Train_X = Corpus['tweet']
    Train_Y = Corpus['gender']
    print(f"3a. Read training data file and turned it into a corpus with [{len(Train_X)}] instances")

    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    print("3b. Encoded corpus labels")

    vectorizer = CountVectorizer(max_features=35000)
    Training_X = vectorizer.fit_transform(Train_X)
    print("3c. Vectorized corpus instances to training instances (fit+transform)")
    # SVM1

#№4# Create and train classifier using data from №3

if option == 1:
    current_classifier = Classifier()
    current_classifier.train(training_data)
    print("4. NB classifier created and trained using tuples.")
elif option == 2:
    # SVM2
    SVM = svm.LinearSVC(dual=False)
    # Uncomment to change SVM model/kernel, linearSVC is prefered
    #SVM = svm.SVC(kernel='rbf', gamma=1)
    SVM.fit(Training_X, Train_Y)
    print("4. SVM classifier created and trained using corpus")
    # SVM2

#№5# Read the gender info text file and create list of User objects (initially assigned Male) needed for testing

user_list = []

with open(f'../Data/Original/{lang.upper()}/{lang}_gender_info.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        user_info = line.rstrip('\n').split()
        current_user = User(user_info[0], user_info[1], "M")
        user_list.append(current_user)

print(f"5. Created user list with length [{len(user_list)}].")

#№6# Read the tweets designated for testing and assign them to Users in the user list

with open(f'../Data/Partitions/test_data_fold_{fold}_{lang}.csv', mode='r', newline='', encoding='utf-8') as f:
    for i, line in enumerate(f):
        columns = line.split('\t')
        for j in user_list:
            if j.handle == columns[5]:

                # Only get first 20 tweets
                if len(j.tweets) < 20:
                    current_content = columns[8]
                    # Preprocessing of test tweets
                    current_content = tokenizer(current_content, lang)

                    # Tweet object handles are useless currently but we may decide to keep the users and their tweets separate in the future
                    current_tweet = Tweet(j.handle, current_content)
                    j.tweets.append(current_tweet)

counter = 0
for c in user_list:
    counter += len(c.tweets)

print(f"6. Read [{i}] tweets and assigned [{counter}] of them to users.")

#№7# Assign gender to Users by classifying all of their Tweets

if option == 1:  
    for i, user in enumerate(user_list):
        # NEW: Pass concatenation option
        user.assigned_gender = current_classifier.classify_user(user_list[i], concat)
elif option == 2:
    #SVM3
    for i, user in enumerate(user_list):
        m_count = 0
        f_count = 0

        if concat == 1:
            # CONCATENATED TWEETS VER
            
            final_text = []
            
            for tweet in user_list[i].tweets:
                final_text += tweet.content

            # Ignore empty users because they break SVM
            if final_text:
                
                Testing_X = vectorizer.transform(final_text)
                predictor = SVM.predict(Testing_X)
                
                # Encoder turns F into 0 and M into 1 due to alphabetic order
                if predictor[0] == 1:
                    result = "M"
                else:
                    result = "F"
                user_list[i].assigned_gender = result
        else:
            # SEPARATE TWEETS VER
            for tweet in user_list[i].tweets:
                
                Test_X = tweet.content
                # Ignore empty tweets because they break SVM
                if not Test_X:
                    break

                Testing_X = vectorizer.transform(Test_X)
                predictor = SVM.predict(Testing_X)

                # Encoder turns F into 0 and M into 1 due to alphabetic order
                if predictor[0] == 1:
                    m_count += 1
                else:
                    f_count += 1

            if m_count >= f_count:
                result = "M"
            else:
                result = "F"
            user_list[i].assigned_gender = result
    #SVM3

print("7. All users classified.")

#№8# Evaluate results

current_evaluator = Evaluator(user_list)
current_evaluator.evaluate()
print("8. Evaluator created and calculations complete:")
current_evaluator.results()
