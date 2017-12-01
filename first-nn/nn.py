# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        # use sigmoid activation
        self.activation_function = lambda x: 1.0 / (1.0 + np.exp(-x))  # Replace 0 with your sigmoid calculation.

    def dsigmoid(self, s):
        return s * (1.0 - s)

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        ''' Implement forward pass here

            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        # convert input to 2D array
        features = X.reshape((1, self.input_nodes))

        # =========================
        #  compute the hidden layer
        # =========================
        # convert input to (1,n)
        hidden_inputs = features.transpose()
        # multiply by weights
        hidden_inputs = self.weights_input_to_hidden * hidden_inputs
        # sum the hidden nodes
        hidden_inputs = hidden_inputs.sum(axis=0)
        # reshape to (n,1)
        hidden_inputs = hidden_inputs.reshape((self.hidden_nodes, 1))
        # apply activation function
        hidden_outputs = self.activation_function(hidden_inputs)

        # =====================
        #  compute output layer
        # =====================
        # multiply by weights
        final_inputs = self.weights_hidden_to_output * hidden_outputs
        # sum the output node
        final_outputs = final_inputs.sum(axis=0)

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # Output error
        error = (y - final_outputs)  # Output layer error is the difference between desired target and actual output.

        # hidden layer's contribution to the error
        hidden_error = (hidden_outputs * (1.0 - hidden_outputs))

        # Backpropagated error terms
        output_error_term = error * hidden_outputs
        hidden_error_term = error * self.weights_hidden_to_output * hidden_error

        # create a (3,2) array from hidden_error_term * X[i]
        # that matches the shape of the delta_weights_i_h
        # I couldn't figure out how to do this with numpy array operations
        x = []
        for i in range(len(X)):
            x.append((hidden_error_term * X[i]).reshape(2))
        x = np.array(x)

        # Weight step (input to hidden)
        delta_weights_i_h += x

        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        for i in range(n_records):
            self.weights_hidden_to_output += delta_weights_h_o * self.lr  # update hidden-to-output weights with gradient descent step
            self.weights_input_to_hidden += delta_weights_i_h  * self.lr # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####

        # reshape to 1d array
        ft = features.reshape(features[0].shape)

        # =========================
        #  compute the hidden layer
        #  by procedural loops
        # =========================
        # multiply by weights
        hn = np.zeros(self.hidden_nodes)
        wx = np.zeros(self.weights_input_to_hidden.shape)
        for i in range(self.hidden_nodes):
            for j in range(self.input_nodes):
                f = ft[j]
                w = self.weights_input_to_hidden[j, i]
                wx[j, i] = f * w
        # sum the inputs to the hidden nodes
        for i in range(self.hidden_nodes):
            for j in range(self.input_nodes):
                hn[i] += wx[j, i]
        # apply activation function
        for i in range(self.hidden_nodes):
            hn[i] = self.activation_function(hn[i])
        # reshape to (n,1)
        hn = hn.reshape((self.hidden_nodes, 1))

        # =========================
        # compute the hidden layer
        # =========================
        # reshape input to (1,n)
        hidden_inputs = features.transpose()
        # multiply by weights
        hidden_inputs = self.weights_input_to_hidden * hidden_inputs
        # sum the hidden nodes
        hidden_inputs = hidden_inputs.sum(axis=0)
        # reshape to (n,1)
        hidden_inputs = hidden_inputs.reshape((self.hidden_nodes, 1))
        # apply the activation function
        hidden_outputs = self.activation_function(hidden_inputs)

        # =========================
        # compute the output layer
        # =========================
        # multiply by weights
        final_inputs = self.weights_hidden_to_output * hidden_outputs
        # sum the output
        final_outputs = final_inputs.sum(axis=0)
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1

data_path = '../../deep-learning/first-neural-network/Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)
print(rides.head())

rides[:24 * 10].plot(x='dteday', y='cnt', grid=True)

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
print(data.head())

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean) / std

# Save data for approximately the last 21 days
test_data = data[-21 * 24:]

# Now remove the test data from the data set
data = data[:-21 * 24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

print(test_features.shape)

# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60 * 24], targets[:-60 * 24]
val_features, val_targets = features[-60 * 24:], targets[-60 * 24:]


#############
# In the my_answers.py file, fill out the TODO sections as specified
#############

def MSE(y, Y):
    return np.mean((y - Y) ** 2)


import unittest

inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])


class TestMethods(unittest.TestCase):
    ##########
    # Unit tests for data loading
    ##########

    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == '../../deep-learning/first-neural-network/bike-sharing-dataset/hour.csv')

    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))

    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1 / (1 + np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output,
                                    np.array([[0.37275328],
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[0.10562014, -0.20185996],
                                              [0.39775194, 0.50074398],
                                              [-0.29887597, 0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))


suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)
