# -*- coding: utf-8 -*-
"""
@author: aleja
"""
#Student Number: 201920969

import matplotlib.pyplot as plt 
import numpy as np

from Perceptron import Perceptron




#PART 1
#p = Perceptron(3)
#logic_input = []
#logic_input.append(np.array([1, 0, 0]))
#logic_input.append(np.array([1, 0, 1]))
#logic_input.append(np.array([1, 1, 0]))
#logic_input.append(np.array([1, 1, 1]))

#logic_label = [0,0,0,1]

#print(p.predict(logic_input[2]))
#p.test(logic_input,logic_label)
#print(p.train(logic_input,logic_label))
#p.test(logic_input,logic_label)


#PART 2
# p = Perceptron(785)
# data_path = "./"
# train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
# test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

# target_digit = 7
# train_input = [ np.append([1],d[1:]) for d in train_data ]
# train_label = [ int(d[0]==target_digit) for d in train_data ]

# test_input = [ np.append([1],d[1:]) for d in test_data ]
# test_label = [ int(d[0]==target_digit) for d in test_data ]


# p.test(test_input,test_label)
# p.train(train_input,train_label)
# p.test(test_input,test_label)


# fig = plt.figure(figsize=(4,4))
# data = p.weights[1:].reshape(28,28)
# vis = train_input[0][1:].reshape(28,28)
# plt.imshow(vis)
# plt.show()

#PART 3

#p = Perceptron(785)
#data_path = "./"
#train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
#test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

#target_digit = 7
#train_input = [ np.append([1],d[1:]) for d in train_data ]
#train_label = [ int(d[0]==target_digit) for d in train_data ]

#test_input = [ np.append([1],d[1:]) for d in test_data ]
#test_label = [ int(d[0]==target_digit) for d in test_data ]


#p.test(test_input,test_label)
#p.train_batch(train_input,train_label)
#p.test(test_input,test_label)


#fig = plt.figure(figsize=(4,4))
#data = p.weights[1:].reshape(28,28)
#vis = train_input[0][1:].reshape(28,28)
#plt.imshow(vis)
#plt.show()
