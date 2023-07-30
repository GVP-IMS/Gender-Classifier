### EVALUATION MODULE OVERVIEW ###
# Contains an evaluator class with methods for evaluating a list of users
# Also has a method for printing out results

# Import Space

from Corpus import User

# An instance of this class is used to evaluate the performance of our model using combined F1 score of 2 classes (male/female)
class Evaluator:
    def __init__(self, user_list):
        self.user_list = user_list
        
        self.correct_male = 0
        self.false_male = 0
        self.correct_female = 0
        self.false_female = 0

        self.m_precision = None
        self.m_recall = None
        self.f_precision = None
        self.f_recall = None
        self.f1 = None

    # Looks at the list of users and evaluates the difference between their true and assigned gender, memorizing precision, recall and F1 score
    def evaluate(self):
        for user in self.user_list:
            if user.true_gender == "M":
                if user.true_gender == user.assigned_gender:
                    self.correct_male += 1
                else:
                    self.false_female += 1
            else:
                if user.true_gender == user.assigned_gender:
                    self.correct_female += 1
                else:
                    self.false_male += 1

        # Correct any zeroes to prevent errors
        if self.correct_male == 0:
            self.correct_male = 0.1
        if self.false_female == 0:
            self.false_female = 0.1
        if self.correct_female == 0:
            self.correct_female = 0.1
        if self.false_male == 0:
            self.false_male = 0.1

        m_total = self.correct_male + self.false_female
        f_total = self.correct_female + self.false_male

        self.m_precision = self.correct_male / (self.correct_male + self.false_male)
        self.m_recall = self.correct_male / (m_total)
        m_f1 = (2 * self.m_precision * self.m_recall)/(self.m_precision + self.m_recall)

        self.f_precision = self.correct_female / (self.correct_female + self.false_female)
        self.f_recall = self.correct_female / (f_total)
        f_f1 = (2 * self.f_precision * self.f_recall)/(self.f_precision + self.f_recall)

        self.f1 = ( (m_f1 * m_total) + (f_f1 * f_total) ) / ( m_total + f_total )

    # Return memorized evaluation score
    def results(self):
        print(f"Combined F1 Score: {self.f1}")
        print(f"Male Precision: {self.m_precision}")
        print(f"Male Recall: {self.m_recall}")
        print(f"Female Precision: {self.f_precision}")
        print(f"Female Recall: {self.f_recall}")
