import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

class Perceptron:
    #Constructor definition
    def __init__(self, eta: float=None, epochs: int=None):
        self.weights = np.random.randn(3) * 1e-4 #getting small random weights
        training = (eta is not None) and (epochs is not None)
        if training:
            print(f'Initial weights is {self.weights}')
        self.eta = eta
        self.epochs = epochs
    
    #_Z_outcome, the net weighted sum, a private method in the class, that starts with an underscore
    def _z_outcome(self, inputs, weights):
        return np.dot(inputs, weights)
    
    #Activation/Decision Function, that takes 'z' outcome
    def activation_function(self, z):
        return np.where(z>0, 1, 0)
    
    #The fit method to fit and perform the training
    def fit(self, X, y):
        self.X = X
        self.y = y
        #X with bias
        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
        
        for epoch in range(self.epochs):
            print('---' * 20)
            print(f'epoch: {epoch + 1} / {self.epochs}')
            print('---' * 20)
            
            z = self._z_outcome(X_with_bias, self.weights)
            y_hat = self.activation_function(z)
            
            print(f'Predicted value after forward pass is: {y_hat}')
            
            #calculating error
            self.error = self.y - y_hat
            print(f'error at epoch {epoch + 1} is {self.error}')
            
            #Updating weights
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)
            print(f'Updated weights after {epoch}/{self.epochs} is {self.weights}')
            print(f'##' * 20)
    
    #The predict method, to predict an unseen data
    def predict(self, test_input):
        X_with_bias = np.c_[test_input, -np.ones((len(test_input),1))]
        z = self._z_outcome(X_with_bias, self.weights)
        return self.activation_function(z)
    
    #We will love to compute the total loss
    def total_loss(self):
        loss = np.sum(self.error)
        print(f'The total loss {loss}')
        #Optional, you can return the loss, if need be
        return loss
    
    #Create the model path and return the path to the model, a private method
    def _create_return_path(self, model_dir, filename):
        os.makedirs(model_dir, exist_ok=True)
        dir_name = os.path.join(model_dir, filename)
        return dir_name
    
    #Save model
    def save(self, filename, model_dir = None):
        if model_dir is not None:
            model_file_path = self._create_return_path(model_dir, filename)
            joblib.dump(self, model_file_path)
        else:
            model_file_path = self._create_return_path('model', filename)
            joblib.dump(self, model_file_path)
    
    #Load model
    def load(self, filepath):
        return joblib.load(filepath)
    