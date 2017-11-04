from __future__ import division
import numpy as np
import argparse
import random
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.collocations import *
import timeit
import sys
import pickle
import gzip


class Classifier(object):
    def __init__():
        pass

    def train():
        """
        Override this method in your class to implement train
        """
        raise NotImplementedError("Train method not implemented")

    def inference():
        """
        Override this method in your class to implement inference
        """
        raise NotImplementedError("Inference method not implemented")

    def calculate_accuracy():
        """
        Override this method in your class to implement inference
        """
        raise NotImplementedError("Calculate accuracy method not implemented")


class MLP(Classifier):
    """
    Implement your MLP here
    """
    def __init__(self, input_length, hidden_layer_size, output_size):

        # determine the step ranges 
        input_range = 1.0 / input_length ** (1/2)
        middle_range = 1.0 / hidden_layer_size[0] ** (1/2)
        output_range = 1.0 / output_size ** (1/2)

        self.hidden_layer_size = hidden_layer_size

        # dictionary to keep track of the parameters
        self.parameterDictionary = {}
        # layers of weights - start random for good initialization
        # bias layers - start at 0 for no bias
        # create a weight matrix input_length * hidden_size since the input is 1 * input_length
        self.parameterDictionary['w1'] = np.random.normal(loc = 0, scale = input_range, size = (input_length, hidden_layer_size[0]))
        # bias matrix size is 1 * hidden_size
        self.parameterDictionary['b1'] = np.zeros(hidden_layer_size[0])
        self.parameterDictionary['w2'] = np.random.normal(loc = 0, scale = middle_range, size = (hidden_layer_size[0], output_size))= 2 * np.random.random((hidden_layer_size[0], hidden_layer_size[1])) - 1
        self.parameterDictionary['b2'] = np.zeros(output_size)

        #super(MLP, self).__init__()

    def sigmoid(self, x):
         return (1/(1 + np.exp(-x)))

    def dersigmoid(self, x):
        return x * (1 - x) 

    def inference(self, X):

        o1 = X.dot(self.parameterDictionary['w1']) + self.parameterDictionary['b1']
        # RELU activation function
        a1 = np.maximum(0, o1)


        # second forward pass
        o2 = a1.dot(self.parameterDictionary['w2']) + self.parameterDictionary['b2']
        # sigmoid function
        a2 = self.sigmoid(o2)


        prediction = [1 if i >= 0.5 else 0 for i in a2]

        return prediction

    def calculate_accuracy(self, validation_x, validation_y):
        predictions = self.inference(validation_x)

        accuracy = 0
        for pred, gold in zip(predictions, validation_y):
            if pred == gold:
                accuracy = accuracy + 1

        accuracy = accuracy / len(validation_x)
        return accuracy

    # set the model weights
    def set_parameters(self, best_parameters):
        self.parameterDictionary['w1'] = best_parameters['w1']
        self.parameterDictionary['w2'] = best_parameters['w2']
        #self.parameterDictionary['b1'] = best_parameters['b1']
        #self.parameterDictionary['b2'] = best_parameters['b2']

    def train(self, X, Y, validation_x, validation_y, learning_rate, learning_rate_decay, regularization=0.90, batch_size=700, num_epochs=160):
        
        best_test_accuracy = 0.0
        best_epoch = 1
        # dictionary to store the weights when we get the best test accuracy
        best_parameters = {}

        # fully connected layers 
        for i in range(num_epochs):

            for j in range(int(len(X) / batch_size)):
                # create a mini batch of the data
                indices = np.random.choice(np.arange(X.shape[0]), size=batch_size, replace=True)
                X_data = X[indices]
                Y_data = Y[indices]


                # forward pass
                # for 2d arrays .dot is equivalent to matrix multiplication
                o1 = X_data.dot(self.parameterDictionary['w1']) + self.parameterDictionary['b1']
                # RELU activation function
                a1 = np.maximum(0, o1)

                #a1 *= np.random.binomial([np.ones((len(X_data), self.hidden_layer_size[0]))],1-0.25)[0] * (1.0/(1-0.25))

                # second forward pass
                o2 = a1.dot(self.parameterDictionary['w2']) + self.parameterDictionary['b2']
                a2 = self.sigmoid(o2)

                # backpropagation
                # hidden units -> compute error term based on weighted average of error terms of nodes that use a_i as an input

                # dictionary to keep track of gradients to make it easier
                gradients = {}
                # for each node i in layer l, compute an error to measure how much node was responsible for errors
                # for output node, measuure difference between network's activation and true target value
                # TODO: Fix

                Y_data = np.reshape(Y_data, (len(Y_data), 1))
                error3 = -(Y_data - a2)
                output_error = error3 * self.dersigmoid(a2)
                
                # error = w.T * error from the above layer * activation
                # the output is related to the w2
                # update = error of above layer * input of this layer
                gradients['w2'] = np.dot(a1.T, output_error)
                #gradients['b3'] = np.sum(output_error, axis=0)

                # error is weight at this layer * error of previous layer
                error = np.dot(output_error, self.parameterDictionary['w2'].T)
                # backpropagate through the RELU activation function
                error[a1 <= 0] = 0

                # a1 is the input into w2
                # update = error of above layer * input of this layer
                gradients['w1'] = np.dot(X_data.T, error)
                #gradients['b2'] = np.sum(error, axis=0)
            
                # update the weights
                self.parameterDictionary['w1'] = self.parameterDictionary['w1'] + (-learning_rate * regularization * gradients['w1'])
                self.parameterDictionary['w2'] = self.parameterDictionary['w2'] + (-learning_rate * regularization * gradients['w2'])
                
                # update the bias
                # self.parameterDictionary['b1'] = self.parameterDictionary['b1'] + (-learning_rate * gradients['b1'])
                # self.parameterDictionary['b2'] = self.parameterDictionary['b2'] + (-learning_rate * gradients['b2'])



            accuracy = self.calculate_accuracy(validation_x, validation_y)
            train_accuracy = self.calculate_accuracy(X, Y)

            # decay the learning rate
            learning_rate *= learning_rate_decay

            

            if accuracy > best_test_accuracy:
                best_test_accuracy = accuracy
                best_parameters['w1'] = self.parameterDictionary['w1']
                best_parameters['w2'] = self.parameterDictionary['w2']
                #best_parameters['b1'] = self.parameterDictionary['b1']
                #best_parameters['b2'] = self.parameterDictionary['b2']

                best_epoch = i

            print("Train Accuracy at epoch " + str(i) + ": " + str(train_accuracy))
            print("Test Accuracy at epoch " + str(i) + ":  " + str(accuracy))
        print("Best validation accuracy: " + str(best_test_accuracy))
        print("Best epoch: " + str(best_epoch))

        #with open('best_parameters_dictionary.pickle', 'wb') as handle:
        with gzip.open('best_parameters_dictionary.gzip', 'wb') as f:
            pickle.dump(best_parameters, f)
            #pickle.dump(best_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)



class Perceptron(Classifier):
    """
    Implement your Perceptron here
    """
    def __init__(self, input_length):
        #super(Perceptron, self).__init__()

        # perceptron only has one layer so only one weight vector needed
        #self.weights = np.random.randn(input_length)
        self.weights = np.zeros(input_length)

    def inference(self, x_data):
        prediction_list = []

        for i in range(len(x_data)):
            output = x_data[i].dot(self.weights)
            # make a prediction (1 if output > 0, else 0)
            # prediction = [1 if i > 0 else 0 for i in output]
            prediction = np.sign(output)
            if prediction == -1:
                prediction = 0
            prediction_list.append(prediction)

        return prediction_list

    def calculate_accuracy(self, validation_x, validation_y):
        predictions = self.inference(validation_x)

        accuracy = 0
        for pred, gold in zip(predictions, validation_y):
            if pred == gold:
                accuracy = accuracy + 1

        accuracy = accuracy / len(validation_x)
        return accuracy


    def train(self, X, Y, validation_x, validation_y, learning_rate, num_epochs=50):

        # calculate the output (w * x)
        #output = x_data.dot(self.weights) + self.bias
        # make a prediction (1 if output > 0, else -1)
        # calculate the error
        # update the weights based on the learning rate and error

        # keep track of best accuracy to avoid overfitting
        best_accuracy = 0
        previous_best_accuracy = 0
        third_prev_best_accuracy = 0

        for j in range(num_epochs):
            for i in range(0, len(X)):
                # if y' != y
                multiple = 0
                if Y[i] == 0:
                    multiple = -1
                else:
                    multiple = 1
                if (np.dot(X[i], self.weights)*multiple) <= 0:
                    self.weights = self.weights + learning_rate*multiple*X[i]

            # use the validation set to control overfitting. 
            # stop training if we start to overfit the train set.
            current_accuracy = self.calculate_accuracy(validation_x, validation_y)
            if best_accuracy == previous_best_accuracy == third_prev_best_accuracy == current_accuracy:
                print("Best accuracy: " + str(best_accuracy) + " current accuracy: " + str(current_accuracy))
                print("Stopped training at epoch: " + str(j))
                break
            else:
                third_prev_best_accuracy = previous_best_accuracy
                previous_best_accuracy = best_accuracy
                best_accuracy = current_accuracy


    

def feature_extractor_test_data(data, labels, all_words):

    data_array = np.zeros((len(data), len(all_words)))

    all_words = list(all_words)

    for index, row in enumerate(data):
        tokens = [word.lower() for word in row.split()]
        #words_no_stopwords = [word for word in tokens if not word in stopwordsSet]
        for i, word in enumerate(tokens):
            try:
                word_index = all_words.index(word)
                data_array[index, word_index] = 1
            except:
                continue

    x_data_array = np.asarray(data_array)

    return x_data_array, np.array(labels)
    #return bigram_data_array, np.array(labels)

def feature_extractor_training_data(data, labels):
    """
    implement your feature extractor here
    """

    # stopwords from NLTK
    stopwordsSet = set(stopwords.words("english"))

    # unique words set without stopwords
    unique_words = set()
    all_words = []
    for row in data:
        words = row.split()
        for i in words:
            if i.lower() not in unique_words and i.lower() not in stopwordsSet:
                unique_words.add(i.lower())
                all_words.append(i.lower())

    data_array = np.zeros((len(data), len(all_words)))

    for index, row in enumerate(data):
        tokens = [word.lower() for word in row.split()]
        #words_no_stopwords = [word for word in tokens if not word in stopwordsSet]
        for i, word in enumerate(tokens):
            try:
                word_index = all_words.index(word)
                data_array[index, word_index] = 1
            except:
                continue

    # the bag of words representation
    x_data_array = np.asarray(data_array)

    return x_data_array, np.array(labels), all_words

def evaluate(preds, golds):

    tp, pp, cp = 0.0, 0.0, 0.0
    # zip -> iterator that aggreagates elements from each of the iterables
    for pred, gold in zip(preds, golds):

        if pred == 1:
            pp += 1
        if gold == 1:
            cp += 1
        if pred == 1 and gold == 1:
            tp += 1
    precision = tp / pp
    recall = tp / cp


    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)

def main():

    start_time = timeit.default_timer()

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--best_parameters")
    args, leftovers = argparser.parse_known_args()        

    with open("sentences.txt", encoding='latin-1') as f:
        data = f.readlines()
    with open("labels.txt", encoding='latin-1') as g:
        labels = [int(label) for label in g.read()[:-1].split("\n")]


    """
    Cross Validation - Separate data into 60%, 20%, 20% (training, validation, test)
    Validation - If accuracy of training data set increases but validation doesn't, then stop training
    """
    # return a list of numbers in a random order
    data_length = len(data)

    indices = np.random.permutation(data_length)
    training_index, validation_index, test_index = indices[:int(.6 * data_length)], indices[int(.6 * data_length):int(.8 * data_length)], indices[int(.8 * data_length):]

    data = np.asarray(data)
    labels = np.asarray(labels)

    training_x_data = data[training_index]
    training_y_data = labels[training_index]
    validation_x_data = data[validation_index]
    validation_y_data = labels[validation_index]
    test_x_data = data[test_index]
    test_y_data = labels[test_index]


    """
    Extract features
    """
    if args.best_parameters is None:
        training_x_data, training_y_data, unique_words = feature_extractor_training_data(training_x_data, training_y_data)
        np.save("unique_words.npy", unique_words)
    else:
        unique_words = np.load("unique_words.npy")
        training_x_data, training_y_data = feature_extractor_test_data(training_x_data, training_y_data, unique_words)

    validation_x_data, validation_y_data = feature_extractor_test_data(validation_x_data, validation_y_data, unique_words)
    test_x_data, test_y_data = feature_extractor_test_data(test_x_data, test_y_data, unique_words)

    """
    Initialize the algorithms
    """
    data_sample_length = len(training_x_data[0])
    myperceptron = Perceptron(data_sample_length)

    hidden_layer_size = [12]
    output_size = 1
    mymlp = MLP(data_sample_length, hidden_layer_size, output_size)

    """
    Training
    """
    learning_rate = 0.001
    #learning_rate_finish = 0.0003
    #learning_rate_decay = (learning_rate_finish / learning_rate) ** (1./num_epochs) 
    learning_rate_decay = 1
    num_epochs = 160


    if args.best_parameters is None:
        mymlp.train(training_x_data, training_y_data, validation_x_data, validation_y_data, learning_rate, learning_rate_decay, num_epochs=num_epochs)
    else:
        with gzip.open(args.best_parameters, 'rb') as f:
            best_parameters_dictionary = pickle.load(f)
        mymlp.set_parameters(best_parameters_dictionary)

    myperceptron.train(training_x_data, training_y_data, validation_x_data, validation_y_data, 1.0)



    """
    Testing on testing data set
    """

    #with open("sentences.txt", encoding='latin-1') as f:
    #    test_x = f.readlines()
    #with open("labels.txt", encoding='latin-1') as g:
    #    test_y = np.asarray([int(label) for label in g.read()[:-1].split("\n")])


    predicted_y = mymlp.inference(test_x_data)
    precision, recall, f1 = evaluate(predicted_y, test_y_data)
    #print "MLP results", precision, recall, f1
    accuracy = mymlp.calculate_accuracy(test_x_data, test_y_data)
    print("MLP Accuracy: " + str(accuracy))
    print("MLP results: " + str(precision) + " " + str(recall) + " " + str(f1))
    #test_x, test_y = feature_extractor(data, labels)

    
    predicted_y = myperceptron.inference(test_x_data)
    precision, recall, f1 = evaluate(predicted_y, test_y_data)
    accuracy = myperceptron.calculate_accuracy(test_x_data, test_y_data)
    print("Perceptron Accuracy: " + str(accuracy))
    #print "Perceptron results", precision, recall, f1
    print("Perceptron results" + " " + str(precision) + " " + str(recall) + " " + str(f1))


    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    """
    Testing on unseen testing data in grading
    """
    argparser.add_argument("--test_data", type=str, default="../test_sentences.txt", help="The real testing data in grading")
    argparser.add_argument("--test_labels", type=str, default="../test_labels.txt", help="The labels for the real testing data in grading")

    parsed_args = argparser.parse_args(sys.argv[1:])
    real_test_sentences = parsed_args.test_data
    real_test_labels = parsed_args.test_labels
    with open(real_test_sentences, encoding='latin-1') as f:
        real_test_x = f.readlines()
    with open(real_test_labels, encoding='latin-1') as g:
        real_test_y = [int(label) for label in g.read()[:-1].split("\n")]
        #real_test_y = g.readlines()

    real_test_x, real_test_y = feature_extractor_test_data(real_test_x, real_test_y, unique_words)
    print(real_test_x.shape)
    print(real_test_y.shape)

    predicted_y = mymlp.inference(real_test_x)
    precision, recall, f1 = evaluate(predicted_y, real_test_y)
    print("MLP results: " + str(precision) + " " + str(recall) + " " + str(f1))
    #print "MLP results", precision, recall, f1

    
    predicted_y = myperceptron.inference(real_test_x)
    precision, recall, f1 = evaluate(predicted_y, real_test_y)
    print("Perceptron results" + " " + str(precision) + " " + str(recall) + " " + str(f1))
    #print "Perceptron results", precision, recall, f1



if __name__ == '__main__':
    main()
