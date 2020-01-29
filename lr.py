import csv
import math
import re
import numpy as np
import emoji
import emot
from numpy.linalg import multi_dot

class Linear_Regression:

    def __init__(self,x_train, y_train):
        self.x_train = x_train
        y_train = [ 0 if x == "0\n" else 1 for x in y_train]  
        self.y_train = np.transpose(np.matrix([y_train]))
        # self.distribution_aggresive = {}
        # self.distributive_not_aggresive = {}
        self.final = []
        self.word_dict = {}
        self.document_count()
        # self.document_count()

    def tokenize(self, s):
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
        # Remove punctuation and all weird characters
        s = re.sub("\W"," ", s)
        # Special character cleaning
        s = re.sub("\s"," ", s)
        return list(s.split())

    def document_count(self):
        count_in_doc = {}
        for x in self.x_train:
            for i in self.tokenize(x):
                if i not in count_in_doc:
                    self.word_dict[i] = []
                    count_in_doc[i] = [0]*12000
        count_in_doc["BIAS"]= [1]*12000
        for k in range(len(self.x_train)):
            i_long = self.tokenize(self.x_train[k])
            for i in i_long:
                count_in_doc[i][k] = (i_long.count(i))
        l = []
        for key, value in count_in_doc.items():
            l.append(count_in_doc[key])
        l = np.transpose(np.matrix(l))
        print(l.shape)
        self.final = l 

    #Lets create a simple neuron matrix where the words are the biases and the weights 
    # def create_two_layer_matrix(self):
    def calculate_log_likehood(self):



    def sigmoid(self,x):
        x = np.array(x)
        return 1/(1+np.exp(-1*x))

    def log_likelihood(self, b):
        pass

x_train = open('X_train.txt').readlines()
y_train = open('Y_train.txt').readlines()
x_test = open('X_test.txt').readlines()
Linear_Regression(x_train, y_train)