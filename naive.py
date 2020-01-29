import csv
import math
import re
import numpy as np
import emoji
import emot
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

class NaiveBayes:

    def __init__(self,x_train, y_train):
        self.x_train = x_train
        y_train = [ 0 if x == "0\n" else 1 for x in y_train]  
        self.y_train = y_train
        self.distribution_aggresive = {}
        self.distributive_not_aggresive = {}
        self.final = {}
        self.document_count()

    '''Write a function called tokenize that takes in a string and tokenizes it by whitespace,
    returning a list of tokens. You should use this function as you read 
    in the training data so thateach whitespace separated word will be 
    considered a different feature (i.e., a differentxi). 
    '''
    def tokenize(self, s):
        #Lowercase every string
        return list(s.split())


    '''
    The improvements here are emoji and empoticons are converted to text and weird characters are removed and treated the same
    '''
    def better_tokenize(self, s):
        #Lowercase every string
        s = s.lower()
        #convert emojis to text
        s = emoji.demojize(s, delimiters=("", ""))
        s = emoji.demojize(s, delimiters=("", ""))
        answ = emot.emoticons(s)
        if(str(type(answ))== "<class 'list'>"):
            answ = answ[0]
        if(answ['flag']):
            # s = s.replace(answ['value'],answ['mean'])
            j = 0
            for i in answ['value']:
                s = s.replace(i, " "+ answ['mean'][j].split()[-1])
                j = j+1
        # s = s.replace(ans['value'],ans['mean'])
        # remove punctuation and all weird characters
        s = re.sub("\W"," ", s)
        #Special character cleaning
        s = re.sub("\s"," ", s)
        return list(s.split())

    # Countss the number of documents
    def document_count(self):
        count_in_doc = {}
        for i in self.x_train: 
            i = set(self.better_tokenize(i))
            for j in i:
                if(j in count_in_doc.keys()):
                    count_in_doc[j]  = count_in_doc[j] +1
                else:
                    count_in_doc[j]  = 1
        #Add to the initial final_list
        for i,j in count_in_doc.items():
            self.final[i] = [j]
        return count_in_doc

    # Splits the training data in aggresive and non aggresive tweets
    def split_train_data(self):
        aggesive = []
        not_aggresive = []
        for i in range(len(self.y_train)):
            if(self.y_train[i] == 1):
                aggesive.append(x_train[i])
            else:
                not_aggresive.append(x_train[i])
        aggresive_counts = (self.word_counts(aggesive))
        non_aggresive_counts = (self.word_counts(not_aggresive))
        return(aggresive_counts , non_aggresive_counts)

    # This funtion updates the weights with the TD - IDF score to get the final dictionary 
    def update_final(self, distribution_aggresive_count,distributive_not_aggresive_count ,distribution_aggresive_probabilities, distributive_not_aggresive_probabilities):
        for i in self.final.keys():
            if(i in distribution_aggresive_count):
                #lets balance the score with the tdif  
                self.final[i].append(distribution_aggresive_count[i])
                # self.final[i].append( distribution_aggresive_probabilities[i])
                self.final[i].append(math.log10(10000/self.final[i][0]) * distribution_aggresive_probabilities[i])
            else:
                self.final[i].append(0)
                self.final[i].append(0)
            if(i in distributive_not_aggresive_count):
                self.final[i].append(distributive_not_aggresive_count[i])
                # self.final[i].append( distributive_not_aggresive_probabilities[i])
                # math.log10(8000/self.final[i][0]) is the TD - IDF scaler which takes common words and damps it 
                self.final[i].append(math.log10(10000/self.final[i][0]) * distributive_not_aggresive_probabilities[i])
            else:
                self.final[i].append(0)
                self.final[i].append(0)
   
    # Count the words to normalize the score with TD - IDF offset
    def word_counts(self ,  l, smoothingalpha = 0 ):
        word_counts = {}
        count = 0
        #word counts the count of all the words
        for i in l:
            i = self.better_tokenize(i)
            for j in i:
                count = count +1
                if(j in word_counts.keys()):
                    word_counts[j] = word_counts[j] + 1
                else:
                    word_counts[j] = 1 
        return [word_counts, count ]

    # Calculates the smoothing and probabilites score
    def probability_distributions(self, dict_count , smoothingalpha):
        x = {}
        for i in dict_count[0]:
            x[i] = (dict_count[0][i] + smoothingalpha ) / (dict_count[1]+ smoothingalpha *self.total_count )
        return x

    def train(self, smoothingalpha = 0):
        if(smoothingalpha <0):
            print("smoothing alpha cannot be less than zero")
            raise
        # P(Y= yi) calculation
        self.p_of_not_aggresive = self.y_train.count(0)/len(self.y_train)

        # creating the distribution for aggresive and non aggresive words
        distribution_aggresive_count , distributive_not_aggresive_count = self.split_train_data()
        # Setting the smoothing alpha to be a global variable to be used everywhere 
        self.smoothingalpha = smoothingalpha
        # self.all_words= set(aggresive_counts[0].keys())|set(non_aggresive_counts[0].keys()
        # counting the total words for smoothing 
        self.total_count = len(set(distribution_aggresive_count[0].keys())|set(distributive_not_aggresive_count[0].keys()))
        ## get ratio
        self.count_aggresive_non_aggresive = [distribution_aggresive_count[1] , distribution_aggresive_count[1]]

        ## Calculating the proabbilites with smoothing 
        distribution_aggresive_probabilities = self.probability_distributions(distribution_aggresive_count , smoothingalpha)
        distributive_not_aggresive_probabilities =self.probability_distributions(distributive_not_aggresive_count , smoothingalpha)
        # UpdateFinal calculates the probabilites
        self.update_final(distribution_aggresive_count[0],distributive_not_aggresive_count[0] ,distribution_aggresive_probabilities, distributive_not_aggresive_probabilities)

    def get_probability_values(self, word):
        result = [0,0]
        for i in self.better_tokenize(word):
            if(i in self.final):
                val = self.final[i]
                if(val[1] != 0):
                    result[0] = result[0] + math.log10(val[2])
                else:
                    if(self.smoothingalpha!=0):
                        result[0] = result[0] + math.log10((self.smoothingalpha) / (self.count_aggresive_non_aggresive[0] + self.smoothingalpha*self.total_count))

                if(val[3] != 0):
                    result[1] = result[1] + math.log10(val[4])
                else:
                    if(self.smoothingalpha!=0):
                        result[1] = result[1] + math.log10(self.smoothingalpha / (self.count_aggresive_non_aggresive[1] + self.smoothingalpha*self.total_count  ))
            else:
                result[0] = result[0] + (self.smoothingalpha / (self.count_aggresive_non_aggresive[0] + self.smoothingalpha*self.total_count  ))
                result[1] = result[1] + (self.smoothingalpha / (self.count_aggresive_non_aggresive[1] + self.smoothingalpha*self.total_count  ))
        return result
                    
    # Classify calculates the result of the naive bayes score
    def classify(self, x_test):
        result = []
        # print(self.distribution_aggresive)
        for i in x_test:
            result.append(self.calculate_greater(self.get_probability_values(i), i))
        f = open('numbers2.csv', 'w')
        with f:
            writer = csv.writer(f)
            writer.writerow(["Id", "Category"])
            n= 0 
            for row in result:
                writer.writerow([n ,row])
                n = n+1

    # Calculates the better probabilities and return 0 or 1 
    def calculate_greater(self,result , i ):
        # print(result)
        p_aggreasive = 10** result[0]
        p_not_aggresive =  10** result[1]
        if((p_aggreasive == 0.0) and (p_not_aggresive == 0.0) ):
            return("0")
        n = ((self.p_of_not_aggresive * p_not_aggresive ) + (1.0-self.p_of_not_aggresive)*p_aggreasive)
        final_nb_aggresives = ((1.0-self.p_of_not_aggresive)*p_aggreasive)/ n
        final_nb_not_aggresives = (self.p_of_not_aggresive * p_not_aggresive )/n
        # print(final_nb_aggresives, final_nb_not_aggresives ,  i )
        if(final_nb_aggresives > final_nb_not_aggresives):
            return(1)
        return(0)

    def f1_score_result(self, x_dev, y_dev):
        result = []
        # print(self.distribution_aggresive)
        for i in x_dev:
            result.append(self.calculate_greater(self.get_probability_values(i), i))
        n = (f1_score(y_dev, result, average = 'weighted'))
        print(n)
        return n

    def f1_score_all(self, x_dex,y_dev):
        D = {}
        for i in range(0, 500, 75):
            self.train(i/250)
            D[i/250]= self.f1_score_result( x_dev, y_dev)
        plt.bar(range(len(D)), list(D.values()), align='center')
        plt.xticks(range(len(D)), list(D.keys()))
        print(D)
        plt.savefig('foo.png')




x_train = open('X_train.txt').readlines()
y_train = open('Y_train.txt').readlines()
x_test = open('X_test.txt').readlines()
x_dev= open('X_dev.txt').readlines()
y_dev = open('y_dev.txt').readlines()
y_dev = [ 0 if x == "0\n" else 1 for x in y_dev] 

nb = NaiveBayes(x_train,y_train)
nb.train(1)
nb.classify(x_test)
print("F score : ", nb.f1_score_result(x_dev, y_dev))
nb.f1_score_all(x_dev, y_dev)


