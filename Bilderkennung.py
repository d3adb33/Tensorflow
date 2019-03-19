from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

from plotting import *

# save path
dir_path = "//home//thans//tensorflow_training//"
# load dataset
mnist = input_data.read_data_sets(dir_path+"//data//", one_hot=True)
#mnist = tf.keras.datasets.mnist.load_data(dir_path+"//data//")
train_size = mnist.train.num_examples 
test_size = mnist.test.num_examples

# dataset vars
features = 784 
classes = 10
# model vars
layer_nodes = [features, 300, 500, classes] # input, hidden, output NEURON org. 250
stddev  = 0.100 # std Abweichung
bias_weight_init = 0.0
# training hyper parameters
epochs = 25
train_batch_size = 256 # max 256 [16:256]
test_batch_size = 1000
learning_rate = 0.002

# mini batch global var / helper
train_mini_batches = int(train_size/train_batch_size)+1
test_mini_batches = int(test_size/test_batch_size)+1

train_erros, test_errors = [], []
train_accs, test_accs = [], []

#TF placeholders (input|output)
x = tf.placeholder(dtype=tf.float32, shape=[None, features], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name="y")

#weights and bias
w1 = tf.Variable(tf.truncated_normal([layer_nodes[0], layer_nodes[1]], stddev=stddev), name='W1')
w2 = tf.Variable(tf.truncated_normal([layer_nodes[1], layer_nodes[2]], stddev=stddev), name='W2')
w3 = tf.Variable(tf.truncated_normal([layer_nodes[2], layer_nodes[3]], stddev=stddev), name='W3')
w4 = tf.Variable(tf.truncated_normal([layer_nodes[3], layer_nodes[4]], stddev=stddev), name='W4')

b1 = tf.Variable(tf.constant(bias_weight_init, shape=[layer_nodes[1]]), name = "b1")
b2 = tf.Variable(tf.constant(bias_weight_init, shape=[layer_nodes[2]]), name = "b2")
b3 = tf.Variable(tf.constant(bias_weight_init, shape=[layer_nodes[3]]), name = "b3")
b4 = tf.Variable(tf.constant(bias_weight_init, shape=[layer_nodes[4]]), name = "b4")

def num_weights(layer_nodes, features):
    w = 0
    b = 0
    last_layer = features
    print("\n\nTrainable Parameters: ")
    for i, layer in enumerate(layer_nodes[1:]):
        if i == len(layer_nodes)-1:
            print("Outputlayer: ")
        else:
            print("Hidden Layer:", i+1)
        print("Weights: ", layer * last_layer)
        print("Biases: ", layer)
        w += layer * last_layer
        b += layer
        last_layer = layer
    print("Overall train parameter: ", w+b, "\n\n")

num_weights(layer_nodes, features)

#model Neuro Network
def nn_model(x):
    input_layer_dict = {"weights": w1, "biases": b1} #input to hidden
    hidden_layer_dict = {"weights": w2, "biases": b2} #hidden to output
    #input layer
    input_layer = x
    #input to hidden
    hidden_layer_in = tf.add(tf.matmul(input_layer, input_layer_dict["weights"]), input_layer_dict["biases"]) # W * x + b
    hidden_layer_out = tf.nn.relu(hidden_layer_in)
    #input to output
    output_layer = tf.add(tf.matmul(hidden_layer_out, hidden_layer_dict["weights"]), hidden_layer_dict["biases"]) 
    return output_layer

#train and testing NN
def nn_run():
    pred = nn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)) # because of softmax function no activation required
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_result = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_result, tf.float32)) # 1/n Sum

#start session and set variables otherwise variables are empty
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #training
        for epoch in range(epochs):
            train_acc, train_loss = 0.0, 0.0
            test_acc, test_loss = 0.0, 0.0
            # for loop mini batch
            for i in range(train_mini_batches):
                epoch_x, epoch_y = mnist.train.next_batch(train_batch_size)
                feed_dict = {x: epoch_x, y: epoch_y}
                sess.run(optimizer, feed_dict = feed_dict) #
            #check performance of train set
            for i in range(train_mini_batches):
                epoch_x, epoch_y = mnist.train.next_batch(train_batch_size)
                feed_dict = {x: epoch_x, y: epoch_y}
                a, c = sess.run([accuracy, cost], feed_dict = feed_dict)
                train_acc += a # [0.9 + 0.8 + 0.85]/3
                train_loss += c
            train_acc = train_acc / train_mini_batches
            train_loss = train_loss / train_mini_batches
            #check performance of train set
            for i in range(test_mini_batches):
                epoch_x, epoch_y = mnist.test.next_batch(test_batch_size)
                feed_dict = {x: epoch_x, y: epoch_y}
                a, c = sess.run([accuracy, cost], feed_dict = feed_dict)
                test_acc += a # [0.9 + 0.8 + 0.85]/3
                test_loss += c
            test_acc = test_acc / test_mini_batches
            test_loss = test_loss / test_mini_batches
            print("Epoch: ", epoch+1, " of ", epochs, 
            " - Train loss: ", round(train_acc, 3), " Train Acc: ", round(train_acc, 3), 
            " - Test loss: ", round(test_loss, 3), " Test Acc: ", round(test_acc, 3))

            # append losses and accs to lists
            train_erros.append(train_loss)
            test_errors.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

        #testing
        test_acc, test_loss = 0.0, 0.0
        for i in range(test_mini_batches):
            epoch_x, epoch_y = mnist.test.next_batch(test_batch_size)
            feed_dict = {x: epoch_x, y: epoch_y}
            a, c = sess.run([accuracy, cost], feed_dict = feed_dict)
            test_acc += a # [0.9 + 0.8 + 0.85]/3
            test_loss += c
        test_acc = test_acc / test_mini_batches
        test_loss = test_loss / test_mini_batches
        print("Test Loss: ", round(test_loss, 3), " Test Acc: ", round(test_acc, 3))
        # final display of convergence
        display_convergence_err(train_erros, test_errors)
        display_convergence_acc(train_accs, test_accs)

if __name__ == "__main__":
    nn_run()