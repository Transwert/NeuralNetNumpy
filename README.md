# NeuralNetNumpy

This is a neural network implemented entirely in `numpy` to do sentiment analysis. It is a simple feed forward neural network. It also uses `nltk` to preprocess the data and eliminate stopwords. It is a small network, and can be run on my Macbook pro in less than 5 minutes.

The sentences for the task can be found in `sentences.txt` and the labels are in `labels.txt`. 

This program also implements the perceptron algorithm. 

## Running the Model

To run the network with the best parameters and the saved data, do this. This will be pretty quick (it is on my Macbook Pro):  
```python2.7 neural_net.py --best_parameters='all_best_parameters_dictionary.gzip' --unique_words='all_unique_words.npy' --test_data='sentences.txt' --test_labels='labels.txt'```


In order to train the network and run it, you can run this command, but it will take a bit longer (about 5 minutes for me) since you will train it.  
`python3 hw1.py --test_data='sentences.txt' --test_labels='labels.txt'`
