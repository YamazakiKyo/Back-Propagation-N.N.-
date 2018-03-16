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

def relu(tensor, derivative = False):
    """ Rectified Linear Unit (activation function):
        Feed forward: if x > 0, return f(x); else return 0
        Feed backward: if x > 0, return 1; else return 0 """
    if (derivative == False):
        return np.maximum(tensor, 0, tensor)
    if (derivative == True):
        return np.heaviside(tensor, 0)

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
    num_hidden_1 = 791 # The number of neurons in the 1st hidden layer
    num_hidden_2 = 791 # The number of neurons in the 2nd hidden layer
    num_output = 1 # Only one single neuron in the output layer since it's binary classification
    num_train_data = 209 # The size of the training set
    learning_rate = 0.1
    np.random.seed(0)

    weight = {
        'input_to_h1': np.random.normal(size=(num_hidden_1, num_input)), # 791 X 12288
        'h1_to_h2': np.random.normal(size=(num_hidden_2, num_hidden_1)), # 791 X 791
        'h2_to_output': np.random.normal(size=(num_output, num_hidden_2)) # 1 X 791
    }

    bias = {
        'input_to_h1': np.random.normal(size=(num_hidden_1, num_train_data)),
        'h1_to_h2': np.random.normal(size=(num_hidden_2, num_train_data)),
        'h2_to_output': np.random.normal(size=(num_output, num_train_data))
    }

    return weight, bias, learning_rate

def forward_nextLayer(input_tensor, weight, bias, AF):
    """ Net input to the next layer Y = W * X + b
        Activation of next layer H = g(Y) """
    a = np.dot(weight, input_tensor)
    if AF == 'sigmoid':
        h = sigmoid(a, derivative=False)
        return h, a
    if AF == 'relu':
        h = relu(a, derivative=False)
        return h, a

def forward_propagation(input_tensor, weight, bias):
    """ Input -- relu --> H1 -- relu --> H2 -- sigmoid --> Output (H3) """
    ''' Output in H1, Input to H2 '''
    h1, a1 = forward_nextLayer(input_tensor, weight['input_to_h1'], bias['input_to_h1'], 'sigmoid')  # h1 is a 791 X 209 tensor, the output from the 1st hidden layer
    ''' Output in H2, Input to H3 '''
    h2, a2 = forward_nextLayer(a1, weight['h1_to_h2'], bias['h1_to_h2'], 'sigmoid')  # h2 is a 791 X 209 tensor , the output from the 2nd hidden layer
    ''' Output in H3 '''
    h3, a3 = forward_nextLayer(a2, weight['h2_to_output'], bias['h2_to_output'], 'sigmoid')  # h3 is a 1 X 209 tensor of predicted labels, the predicted label
    return a1, h1, a2, h2, a3, h3

# def weight_decay(learning_rate, delta_weight):
#     learning_rate = learning_rate - 0.5 * delta_weight
#     return learning_rate

def backward_propagation(a1, h1, a2, h2, a3, h3, train_data, train_label, learning_rate, weight, bias):
    """ For the forward direction is i --> j --> k:
        1. delta weight in unit i = -1 * learning rate * dot (error in unit j, activation in unit i)
        2. error in unit j = gradient in unit j * dot (weight in unit j, error in unit k)
        3. error in output unit = -1 * gradient in output unit * (class label - predicted label)"""
    ''' Output Unit '''
    error = h3 - train_label  # size = 1 X 209
    delta_output = -1 * sigmoid(a3, derivative=True) * error  # size = 1 X 209
    bias_h2_output_new = bias['h2_to_output'] - learning_rate * delta_output
    ''' H2 Layer '''
    delta_weight_h2_output = -1 * learning_rate * np.dot(delta_output, h2.T)  # size = 1 X 791
    weight_h2_output_new = weight['h2_to_output'] + delta_weight_h2_output  # Size Match
    ''' H1 Layer '''
    delta_h2 = sigmoid(a2, derivative=True) * np.dot(weight['h2_to_output'].T, delta_output)  # size = 791 X 209
    bias_h1_h2_new = bias['h1_to_h2'] - learning_rate * delta_h2
    delta_weight_h1_h2 = -1 * learning_rate * np.dot(delta_h2, h1.T)  # size = 791 X 791
    new_weight_h1_h2 = weight['h1_to_h2'] + delta_weight_h1_h2  # Size Match
    ''' Input Layer '''
    delta_h1 = sigmoid(a1, derivative=True) * np.dot(weight['h1_to_h2'].T, delta_h2)  # size = 791 X 209
    bias_input_h1_new = bias['input_to_h1'] - learning_rate * delta_h1
    delta_weight_input_h1 = -1 * learning_rate * np.dot(delta_h1, train_data.T)  # size = 791 X 12288
    new_weight_input_h1 = weight['input_to_h1'] + delta_weight_input_h1  # Size Match

    weight_updated = {
        'input_to_h1': new_weight_input_h1,  # 791 X 12288
        'h1_to_h2': new_weight_h1_h2,  # 791 X 791
        'h2_to_output': weight_h2_output_new # 1 X 791
    }

    bias_updated = {
        'input_to_h1': bias_input_h1_new,
        'h1_to_h2': bias_h1_h2_new,
        'h2_to_output': bias_h2_output_new
    }

    return weight_updated, bias_updated


def train_model(data, label, weight, bias, learning_rate):
    a1, h1, a2, h2, a_output, h_output = forward_propagation(data, weight, bias)
    weight_updated, bias_updated = backward_propagation(a1, h1, a2, h2, a_output, h_output, data, label, learning_rate, weight, bias)
    arr_i = []
    arr_error = []
    for i in range(10000):
        a1, h1, a2, h2, a_output, h_output = forward_propagation(data, weight_updated, bias_updated)
        train_error = np.sum(np.absolute(h_output - label))
        if i % 10 == 0:
            print("This is the %d epoch, the training preidict error is %d: " %(i, train_error))
        arr_i.append(i)
        arr_error.append(train_error / 207)
        if (train_error > 1):
            weight_updated, bias_updated = backward_propagation(a1, h1, a2, h2, a_output, h_output, data, label, learning_rate, weight_updated, bias_updated)
        else:
            break
    return weight_updated, bias_updated, arr_i, arr_error

def do_test(data, label, weight, bias):
    a1, h1, a2, h2, a_output, h_output = forward_propagation(data, weight, bias)
    output = h_output > 0.5
    output = output.astype(int)
    test_error = np.sum(np.absolute(output - label))
    error_rate = test_error / 50.0
    accuracy = 1.0 - error_rate
    # print("The error rate in the testset is %f: " % error_rate)
    print("The accuracy is: %f" % accuracy)
    print("Number of %d test images are misclassified." % test_error)
    return output



""" Here we start our code """
train_data, train_label, test_data, test_label, classes = load_dataset()
# Shape: (12288, 209), (1, 209), (12288, 50), (1, 50)
weight_init, bias_init, lr = setup_NN()
weight_trained, bias_trained, num_iteration, error_rate = train_model(train_data, train_label, weight_init, bias_init, lr)
num_iteration = np.asarray(num_iteration)
error_rate = np.asarray(error_rate)
plt.plot(num_iteration, error_rate)
plt.xlabel('Number of Epochs')
plt.ylabel('Error rate in the training set')
plt.show()
print("------------------------------------")
predicted_label = do_test(test_data, test_label, weight_trained, bias_trained)




