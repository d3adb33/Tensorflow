import tensorflow as tf
import numpy as np
from keras.datasets import boston_housing

#dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

train_size = x_train.shape[0]
test_size = x_test.shape[0]

y_train = np.reshape(y_train, (train_size, 1))
y_test = np.reshape(y_test, (test_size, 1))

# dataset vars
features = x_train.shape[1]
target = 1


# model vars
nodes = [features, 20, target] # input, hidden, output NEURON
train_size = x_train.shape[0]
test_size = x_test.shape[0]
epochs = 5000

#TF placeholders (input|output)
x = tf.placeholder(dtype=tf.float32, shape=[None, features], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None, target], name="y")

#weights and bias
w1 = tf.Variable(tf.random_uniform([nodes[0], nodes[1]], -1, 1))
w2 = tf.Variable(tf.random_uniform([nodes[1], nodes[2]], -1, 1))
b1 = tf.Variable(tf.constant(0.0, shape=[nodes[1]]), name = "b1")
b2 = tf.Variable(tf.constant(0.0, shape=[nodes[2]]), name = "b2")

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
    output_layer_in = tf.add(tf.matmul(hidden_layer_out, hidden_layer_dict["weights"]), hidden_layer_dict["biases"]) 
    output_layer_out = output_layer_in
    return output_layer_out

def r_squared(y_true, y_pred):
    # r² [0, 1]
    numerator = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    denominator = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.clip_by_value(tf.subtract(1.0, tf.div(numerator, denominator)), clip_value_min=0.0, clip_value_max=1.0)
    return r2

#train and testing NN
def nn_run():
    pred = nn_model(x)
    cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
    r2 = r_squared(y, pred)

    #rt session and set variables otherwise variables are empty
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #training
        for epoch in range(epochs):
            train_acc, train_loss = 0.0, 0.0
            feed_dict = {x: x_train, y: y_train}
            sess.run(optimizer, feed_dict = feed_dict) #
            #check performance of train set
            c, r = sess.run([cost, r2], feed_dict = feed_dict)
            if epoch % 100 == 0:
                print("Epoch: ", epoch+1, " of ", epochs, " - Train loss: ", round(c, 4), " - Train r²: ", round(r, 4))
#testing
        feed_dict = {x: x_test, y: y_test}
        p, c, r = sess.run([pred, cost, r2], feed_dict = feed_dict)
        print("Test Loss: ", round(c, 4), "Test r²: ", round(r, 4))
        #print("Predictions:\n", p, "\n\n")

nn_run()