### NB MODULE OVERVIEW ###
# Contains the Naive Bayes classifier class, which has methods for training and then classifying 1 tweet or a user

# Import Space

from decimal import Decimal

# An instance of this class is used to classify singular tweet or all tweets of a user class
class Classifier:
    def __init__(self):
        self.m_counter = 0
        self.f_counter = 0
        self.m_dictionary = {}
        self.f_dictionary = {}

    # 1. Training Method: make classifier memorize weights using data
    def train(self, training_data):

        # Count number of M/F tweets and in how many tweets each word appears
        for tweet in training_data:
            if tweet[1] == "M":
                self.m_counter += 1
                for word in tweet[0]:
                    if word not in self.m_dictionary:
                        self.m_dictionary[word] = 1
                    else:
                        self.m_dictionary[word] += 1
            else:
                self.f_counter += 1
                for word in tweet[0]:
                    if word not in self.f_dictionary:
                        self.f_dictionary[word] = 1
                    else:
                        self.f_dictionary[word] += 1

##        ## Uncomment to count difference in top words    
##        sortedy = dict(sorted(self.m_dictionary.items(), key=lambda x:x[1], reverse=True))
##        sortedy2 = dict(sorted(self.f_dictionary.items(), key=lambda x:x[1], reverse=True))
##
##        for k in sortedy:
##            try:
##                ## add abs() or swap sortedy 1 and 2 to see different counts
##                sortedy2[k] = sortedy2[k]-sortedy[k]
##            except:
##                pass
##
##        sortedy3 = dict(sorted(sortedy2.items(), key=lambda x:x[1], reverse=True))
##        counter = 0
##        for k in sortedy3:
##            if counter <= 200:
##                print(f"{k} {sortedy3[k]}")
##                counter += 1

    # 2. Classify Method: give a label to a single tweet using the memorized weights and Naive Bayes
    def classify_tweet(self, tweet):

        m_probabilities = []
        f_probabilities = []
        alpha_factor = 1

        # Calculate the M/F probability for each word with simple Naive Bayes and Laplace Smoothing with an alpha factor
        for word in tweet:
            if word in self.m_dictionary:
                calculation = Decimal((self.m_dictionary.get(word) + alpha_factor) / (self.m_counter + alpha_factor))
                m_probabilities.append(round(calculation, 15))
            else:
                calculation = Decimal((alpha_factor) / (self.m_counter + alpha_factor))
                m_probabilities.append(round(calculation, 15))
            if word in self.f_dictionary:
                calculation = Decimal((self.f_dictionary.get(word) + alpha_factor) / (self.f_counter + alpha_factor))
                f_probabilities.append(round(calculation, 15))
            else:
                calculation = Decimal((alpha_factor) / (self.f_counter + alpha_factor))
                f_probabilities.append(round(calculation, 15))
        
        # Calculating the product the hard way, just in case
        m_total = Decimal(1)
        f_total = Decimal(1)
        for i in m_probabilities:
            m_total = m_total * i
        for i in f_probabilities:
            f_total = f_total * i

        # Add priors
        m_prior = Decimal( self.m_counter / (self.m_counter+self.f_counter) )
        f_prior = Decimal( self.f_counter / (self.m_counter+self.f_counter) )
        m_total = m_total * m_prior
        f_total = f_total * f_prior

        # Check for zeroes, just in case
        if m_total == 0 or f_total == 0:
            classification_result = (float(0.0), "error")
            
        # Compare M/F score (tiebreaker priority given to Male)
        else:
            if m_total >= f_total:
                classification_result = (float(m_total), "Male")
            else:
                classification_result = (float(f_total), "Female")

        return classification_result

    # 3. Mass Classify Method: look at a user's tweets and, using majority (Male priority), return M/F label.
    def classify_user(self, user, concat):
        m_count = 0
        f_count = 0

        if concat == 1:
            final_text = []
            for tweet in user.tweets:
                final_text += tweet.content
            if final_text:
                if self.classify_tweet(final_text)[1] == "Male":
                    return "M"
                else:
                    return "F"
            else:
                return "M"
        else:
            for tweet in user.tweets:
                if self.classify_tweet(tweet.content)[1] == "Male":
                    m_count += 1
                else:
                    f_count += 1

            username = user.handle
            if m_count >= f_count:
                return "M"
            else:
                return "F"
