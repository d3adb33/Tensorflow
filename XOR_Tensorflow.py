import tensorflow as tf
import numpy as np

def get_dataset():
    x = np.array([[0,0], [1,0], [0,1], [1,1]]).astype(np.float32)
    y = np.array([[0], [1], [1], [0]]).astype(np.float32)
    return x, y

x, y = get_dataset()
x_train, y_train = x, y
x_test, y_test = x, y

# dataset vars

features = 2 
classes = 2
target = 1

# model vars
nodes = [features, 50, target] # input, hidden, output NEURON
train_size = x_train.shape[0]
test_size = x_test.shape[0]
epochs = 1000

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
    output_layer_out = tf.nn.sigmoid(output_layer_in)
    return output_layer_out

#train and testing NN
def nn_run():
    pred = nn_model(x)
    cost = tf.reduce_mean(tf.square(pred - y)) # 1/N Sum (y_hat_n - y_n)²
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
    correct_result = tf.equal(tf.cast(tf.greater(pred, 0.5), tf.float32), y) # val = 0.7 if val > 0.5 then class 1 else class 0
    accuracy = tf.reduce_mean(tf.cast(correct_result, tf.float32)) # 1/n Sum
#start session and set variables otherwise variables are empty
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("weights before training:\n")
        print(sess.run(w1))
        print(sess.run(b1))
        #training
        for epoch in range(epochs):
            train_acc, train_loss = 0.0, 0.0
            feed_dict = {x: x_train, y: y_train}
            sess.run(optimizer, feed_dict = feed_dict) #
            #check performance of train set
            a, c = sess.run([accuracy, cost], feed_dict = feed_dict)
            if epoch % 100 == 0:
                print("Epoch: ", epoch+1, " of ", epochs, " - Train loss: ", round(c, 3), " Train Acc: ", round(a, 3))
#testing
        feed_dict = {x: x_test, y: y_test}
        p, a, c = sess.run([pred, accuracy, cost], feed_dict = feed_dict)
        print("Test Loss: ", round(c, 3), " Test Acc: ", round(a, 3))
        print("Predictions:\n", p, "\n\n")
        print("weights after training:\n")
        print(sess.run(w1))
        print(sess.run(b1))


nn_run()