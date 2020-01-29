import csv
import math
import re
import numpy as np
import emoji
import emot
import random
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score


class LR_v1:

    # Change this value to change the learning rate
    learning_rate = 5*(10**(-5))
    # Steps to do stocastic Gradient descent by 
    steps = 100000

    '''
    This funtion intializes and sets the global variables for the whole class
    '''
    def __init__(self, x_train, x_test, y_train):
        self.x_train = x_train
        self.y_train = np.matrix([y_train ])
        self.y_train = np.transpose(self.y_train)
        
        self.x_test = x_test
        self.X_dict = {}
        self.X = self.document_count()
        self.weights = self.weights()
        self.training_data_log_likehood = []
        # print(self.X.shape, self.)

    # This makes a weights matrix to calculate train the matrix of layer 1 weights
    # this function sets the result to all zeros
    def weights(self): 
        number_of_connections = self.X.shape[1]
        weights =  np.zeros( number_of_connections)
        # Un comment this for random non zero values on a normal distribution with median 1 
        # layer1_weights = np.random.normal(size = final.shape[1]) 
        return np.transpose(np.matrix(weights))

    # Tokenize funtion breaks the string into smalled word bits which can be used for the following steps to better understand the string
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

    def sigmoid(self,x):
        return 1/(1+np.exp(-1*x))

    def sigmoid_vec(self,x):
        return np.vectorize(self.sigmoid)(x)

    def log_likelihood( self):
        x = np.sum(np.multiply(self.y_train , (self.X * self.weights))) - np.sum(np.log(1+ np.exp((self.X * self.weights ))))
        self.training_data_log_likehood.append(x)
        return x 

    def zero_or_one(self,x):
        if(x>(0.50)):
            return 1 
        return 0

    def zero_or_one_vec(self,x):
        return np.vectorize(self.zero_or_one)(x)

    def gradient_cal(self):
        y_result = self.y_train - self.zero_or_one_vec(self.sigmoid_vec(self.X * self.weights  ))
        return np.transpose(self.X)*y_result
    
    def stochastic_gradient_cal(self, n ):
        y_result = self.y_train[n] - self.zero_or_one_vec(self.sigmoid_vec(self.X[n] * self.weights  ))
        return np.transpose(self.X[n])*y_result

    # This function reads through the training data and populates a weight matrix shich storing the dictonary with weights in the self.X_dict variable 
    def document_count(self):
        count_in_doc = {}
        for x in self.x_train:
            for i in self.tokenize(x):
                if i not in count_in_doc:
                    count_in_doc[i] = [0]*12000
        count_in_doc["BIAS"]= [1]*12000
        for k in range(len(x_train)):
            i_long = self.tokenize(self.x_train[k])
            for i in i_long:
                count_in_doc[i][k] = (i_long.count(i))
        l = []
        for key, value in count_in_doc.items():
            l.append(count_in_doc[key])
        l = np.transpose(np.matrix(l))
        # print(l.shape)
        self.X_dict = count_in_doc
        return l
    
    def logistic_regression(self, learning_rate = 5*(10**(-5)) ,steps  =  100000):
        for i in range(steps):
            random_number = random.randint(0,self.X.shape[0]-1)
            n =self.stochastic_gradient_cal(random_number)
            self.weights = self.weights + learning_rate* n
            if(i%1000 == 0):
                print(i, self.log_likelihood())

    def predict(self, x_test):
        temp_count_in_doc = {}
        size = len(x_test)
        for key, value in self.X_dict.items():
            temp_count_in_doc[key] = [0]* size
        temp_count_in_doc["BIAS"] = [1]* size
        for k in range(len(x_test)):
            i_long = self.tokenize(x_test[k])
            for i in i_long:
                if(i in temp_count_in_doc):
                    temp_count_in_doc[i][k] = (i_long.count(i))
        # print(temp_count_in_doc)
        l = []
        for key, value in temp_count_in_doc.items():
            l.append(temp_count_in_doc[key])
        l = np.transpose(np.matrix(l))
        # print(l.shape, self.weights.shape)
        return self.zero_or_one_vec(self.sigmoid_vec(l  * self.weights  ))

    def predict_write_to_file(self,x_test):
        result = result = np.transpose(self.predict(x_test)).tolist()[0]
        f = open('numbers_lr_v3.csv', 'w')
        with f:
            writer = csv.writer(f)
            writer.writerow(["Id", "Category"])
            n= 0 
            for row in result:
                writer.writerow([n ,row])
                n = n+1

    def f_score_one(self, x_dev, y_dev):
        result = self.predict(x_dev)
        n = (f1_score(y_dev, result, average = 'weighted'))
        print(n)

x_train = open('X_train.txt').readlines()
y_train = open('Y_train.txt').readlines()
y_train = [ 0 if x == "0\n" else 1 for x in y_train] 
x_test = open('X_test.txt').readlines()
x_dev= open('X_dev.txt').readlines()
y_dev = open('y_dev.txt').readlines()
y_dev = [ 0 if x == "0\n" else 1 for x in y_dev] 
lr = LR_v1(x_train,x_test, y_train)
lr.logistic_regression()
lr.f_score_one( x_dev, y_dev)
lr.predict_write_to_file(x_test)
