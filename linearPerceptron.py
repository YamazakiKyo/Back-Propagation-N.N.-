import numpy as np
import matplotlib.pyplot as plt
import h5py
import random


def load_dataset():
    """ Load the data from the .h5 file, slice them into training data and testing data (format as 'data | label') """
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_data = np.array(train_dataset["train_set_x"][:])  # Training images (209 x 64 x 64 x 3)
    train_label = np.array(train_dataset["train_set_y"][:])  # Training labels (209 x 1)

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_data = np.array(test_dataset["test_set_x"][:])  # Test data (50 x 64 x 64 x 3)
    test_label = np.array(test_dataset["test_set_y"][:])  # Test labels (50 x 1)

    " Name of the class label, where '0' = 'non-cat' and '1' = 'cat' "
    classes = np.array(test_dataset["list_classes"][:])

    " Transpose to row vecotr "
    train_label = train_label.reshape((1, train_label.shape[0])) # (1 x 209)
    test_label = test_label.reshape((1, test_label.shape[0])) # (1 x 50)

    " Expend the 4 dimension tensors (or you can say 3 dimension tensors in each line) to the higher-dimensional vectors "
    train_data_flatten = train_data.reshape(train_data.shape[0], -1).T
    test_data_flatten = test_data.reshape(test_data.shape[0], -1).T

    " The R, G, and B value for each pixel ranges [0, 255] "
    train_data_norm = train_data_flatten / 255.0
    test_data_norm = test_data_flatten / 255.0

    return train_data_norm, train_label, test_data_norm, test_label, classes

def sigmoid(tensor, derivative = False):
    tensor = np.clip(tensor, -500, 500)
    temp = 1/(1 + np.exp(-1 * tensor))
    if derivative == False:
        return temp
    if derivative == True:
        return tensor * (1 - tensor)

def setup_NN():
    """ Setup the N.N. with initialized parameters """
    num_input = 12288 # The number of neurons in the input layer, or, the size of the feature space from the training set
    num_output = 1 # Only one single neuron in the output layer since it's binary classification
    num_train_data = 209 # The size of the training set
    learning_rate = 0.1
    np.random.seed(0)
    weight_in_out = np.random.normal(size=(num_output, num_input)) #1 X 12288
    return weight_in_out, learning_rate

def forward_propagation(input_tensor, weight):
    """ Net input to the next layer Y = W * X + b
        Activation of next layer H = g(Y)
        Input -- sigmoid --> Output """
    a_out = np.dot(weight, input_tensor)
    h_out = sigmoid(a_out,  derivative=False)
    return a_out, h_out

def backward_propagation(train_data, train_label, a, h, weight, learning_rate):
    error = h - train_label  # size = 1 X 209
    delta_output = -1 * sigmoid(a, derivative=True) * error  # size = 1 X 209
    delta_weight = -1 * learning_rate * np.dot(delta_output, train_data.T)
    weight_updated = weight + delta_weight
    return weight_updated

def train_model(data, label, weight, learning_rate):
    a_output, h_output = forward_propagation(data, weight)
    weight_updated = backward_propagation(data, label, a_output, h_output, weight, learning_rate)
    arr_i = []
    arr_error = []
    for i in range(10000):
        a_output, h_output = forward_propagation(data, weight_updated)
        train_error = np.sum(np.absolute(h_output - label))
        if i % 10 == 0:
            print("This is the %d epoch, the training preidict error is %d: " %(i, train_error))
        arr_i.append(i)
        arr_error.append(train_error / 207)
        if (train_error > 1):
            weight_updated = backward_propagation(data, label, a_output, h_output, weight, learning_rate)
        else:
            break
    return weight_updated, arr_i, arr_error

def do_test(data, label, weight):
    a_output, h_output = forward_propagation(data, weight)
    output = h_output > 0.5
    output = output.astype(int)
    test_error = np.sum(np.absolute(output - label))
    error_rate = test_error / 50.0
    accuracy = 1.0 - error_rate
    # print("The error rate in the testset is %f: " % error_rate)
    print("The accuracy is: %f" % accuracy)
    print("Number of %d test images are misclassified." % test_error)
    # return output

""" Here we start our code """
train_data, train_label, test_data, test_label, classes = load_dataset()
weight_init, lr = setup_NN()
weight_trained, num_iteration, error_rate = train_model(train_data, train_label, weight_init, lr)
num_iteration = np.asarray(num_iteration)
error_rate = np.asarray(error_rate)
plt.plot(num_iteration, error_rate)
plt.xlabel('Number of Epochs')
plt.ylabel('Error rate in the training set')
plt.show()
print("------------------------------------")
do_test(test_data, test_label, weight_trained)