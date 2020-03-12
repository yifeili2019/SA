#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:05:23 2020

@author: emelaktas
"""
# import the numpy library
import numpy as np # to plot error

# matplotlib for visualisation
import matplotlib.pyplot as plt

# pandas for data analysis
import pandas as pd

# the activation function
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

# need to minimise the error - gradient descent, need to find the derivative 
# of sigmoid
def sigmoid_derivative(x): 
    return x * (1.0 - x)

# define a neural network
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4) 
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(self.y.shape)
    
    # Feedforward, activation function    
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1)) 
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
    
    # Backpropagation, derivative of activation function
    def backprop(self):
        # apply the chain rule to find the derivative of the loss function 
        # with respect to weights
        
        # weights between the hidden layer and the output layer
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * 
                                            sigmoid_derivative(self.output)))
        
        # weights between the input layer and the hidden layer
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * 
                                           sigmoid_derivative(self.output), 
                                           self.weights2.T) *
                                           sigmoid_derivative(self.layer1)))
        # update weights
        self.weights1 += d_weights1
        self.weights2 += d_weights2        
        


# input array
X = np.array([[0,0,1], 
              [0,1,1], 
              [1,0,1],
              [1,1,1]])

# output vector
y = np.array([[0],[1],[1],[0]])

# generate a neural network with the function we defined
nn = NeuralNetwork(X,y)

for i in range(1500): 
    nn.feedforward() 
    nn.backprop()
print("nn.output:")
print(nn.output)



# try a different example: notice y has changed
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
# new y
y = np.array([[0],[1],[0],[1]])


nn2 = NeuralNetwork(X,y)

error = []       

for i in range(1500):
    nn2.feedforward()
    nn2.backprop()
    error.append(sum((y - nn2.output)**2))
print("nn2.output:")
print(nn2.output)


# error progression over epochs
plt.plot(error)
plt.xlabel("epoch")
plt.ylabel("error")
plt.savefig('error.png')
plt.show()
 
# first five observations of error
error[:5]

# last five observations of error
error[1495:]
 
# using keras for building neural networks
from keras.models import Sequential
model = Sequential()

from keras.layers import Dense
# Layer 1
model.add(Dense(units=4, activation='sigmoid', input_dim=3))
# Output Layer
model.add(Dense(units=1, activation='sigmoid'))

model.summary()

from keras import optimizers
sgd = optimizers.SGD(lr=1)
model.compile(loss='mean_squared_error', optimizer=sgd)

# Fixing a random seed ensures reproducible results
np.random.seed(9)

X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],[1],[1],[0]])

model.fit(X, y, epochs=1500, verbose=False)

model.predict(X)
print('finish')
