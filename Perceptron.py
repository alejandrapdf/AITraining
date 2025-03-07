import numpy as np


class Perceptron(object):

    #==========================================#
    # The init method is called when an object #
    # is created. It can be used to initialize #
    # the attributes of the class.             #
    #==========================================#
    def __init__(self, no_inputs, max_iterations=20, learning_rate=0.1):
        self.no_inputs = no_inputs
        self.weights = (2*np.random.random(no_inputs)-1)/float(255)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    #=======================================#
    # Prints the details of the perceptron. #
    #=======================================#
    def print_details(self):
        print("No. inputs:\t" + str(self.no_inputs))
        print("Max iterations:\t" + str(self.max_iterations))
        print("Learning rate:\t" + str(self.learning_rate))

    #=========================================#
    # Performs feed-forward prediction on one #
    # set of inputs.                          #
    #=========================================#
    def predict(self, inputs):        
        s = np.dot(inputs,self.weights) + inputs[0]
        if s <= 0:
            return 0
        return 1

    #======================================#
    # Trains the perceptron using labelled #
    # training data.                       #
    #======================================#
    def train(self, training_data, labels):
        assert len(training_data) == len(labels)
        for j in range(self.max_iterations):
            for i in range(len(training_data)):
                if self.predict(training_data[i]) != labels[i]:
                    self.weights = np.add(self.weights,((self.learning_rate*(labels[i]-self.predict(training_data[i])))*training_data[i]))
        return self.weights

    #=========================================#
    # Tests the prediction on each element of #
    # the testing data.
    #=========================================#
    def test(self, testing_data, labels):
        assert len(testing_data) == len(labels)
        c = 0
        for j in range(len(testing_data)):
            #print("actual",labels[j],"estimate",self.predict(testing_data[j]))
            if self.predict(testing_data[j]) == labels[j]:
                c += 1
        
        accuracy = (c/len(testing_data))*100
        print("Accuracy:\t"+str(accuracy))


    def train_batch(self, training_data, labels):
        
        for i in range(self.max_iterations):
            total_samples = 0
            updates = np.zeros(len(self.weights)) 
            for d in training_data: 
                prediction = self.predict(d)
                updates = np.add(updates,((self.learning_rate*(labels[i]-prediction))*training_data[i]))
                total_samples += 1
                
            self.weights = np.add(self.weights,(updates / total_samples))
