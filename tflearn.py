'''
How would you go about building a neural network in TensorFlow? Let's walk through a simple example using data from the MNIST handwritten digit database. We will build a multilayer perceptron network to classify handwritten digits first using pure TensorFlow and then using the TFLearn API. Our features will be the greyscale values of the pixels in our 28 x 28 images.  
'''

#To begin, we import TensorFlow and TFLearn and grab the data from TFLearn's datasets.

import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist

Xtrain, Ytrain, Xtest, Ytest = mnist.load_data(one_hot=True)

#We also set some parameters
N_PIXELS = 28 * 28
N_CLASSES = 10
HIDDEN_SIZE = 64
EPOCHS = 20 

#---------------------- Pure TensorFlow --------------------------------#
sess = tf.Session()

# We define a helper function called "initializer" to initialize our weights with values drawn from a truncated normal distribution (centered around zero).
def initializer(shape):
    return tf.truncated_normal(shape, stddev=shape[0]**-0.5)

# Our model with be trained on data that is fed into our network via placeholders.
x = tf.placeholder(tf.float32, [None, N_PIXELS], name="pixels")
y_label = tf.placeholder(tf.float32, [None, N_CLASSES], name="labels")

# Our weights and biases are stored in variables, which are updated at each training step. The weight matrix has a size equivalent to the number of inputs (or features) by the number of neurons. 
W1 = tf.Variable(initializer([N_PIXELS, HIDDEN_SIZE]), name="weights1")
b1 = tf.Variable(tf.zeros([HIDDEN_SIZE]), name="biases1")

# We multiply the weight matrix by our pixel values, add a bias term, and send the result through a sigmoid activation function. 
hidden = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

# We create a new set of weights and biases for the final layer of our neural network. The weight matrix has a size equal to the number of neurons in the previous layer by the number of digits, 10. 
W2 = tf.Variable(initializer([HIDDEN_SIZE, N_CLASSES]), name="weights2")
b2 = tf.Variable(tf.zeros([N_CLASSES]), name="biases2")

y = tf.matmul(hidden, W2) + b2

# We are solving a multiclass classification problem, so we pass the results from the previous line through a softmax activation function and compute the cross entropy. 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_label))

# We find the weights and biases that minimize our loss via gradient descent. TensorFlow knows that it needs to optimize any tf.Variable associated with "loss."
sgd = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# We also keep track of how our model's accuracy changes during training.
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1)), tf.float32))

# Finally, we initialize our variables and train our model, keeping track of how our test accuracy and loss changes during training.
sess.run(tf.global_variables_initializer())

for i in xrange(EPOCHS):
    sess.run(sgd, feed_dict={x: Xtrain, y_label: Ytrain})
    
    print sess.run([loss, acc], feed_dict={x: Xtest, y_label: Ytest})
    
sess.close()

#---------------------- TFLearn API --------------------------------#

# TFLearn is an API that abstracts away much of TensorFlows verbose syntax, making it easy to read and faster to write code. 

sess = tf.Session()

# With TFLearn, we don't have to worry about creating TensorFlow placeholder and variable operations. Instead of feeding our data into a placeholder, we use TFlearn's "input_data" function. 
x = tflearn.input_data(shape=[None, N_PIXELS], name="pixels")

# It is also simple to create layers using TFLearn. We can create a fully connected layer by feeding the layer's input, number of neurons, and activation function into TFLearn's "fully_connected" function. 
hidden = tflearn.fully_connected(x, HIDDEN_SIZE, activation="sigmoid")
y = tflearn.fully_connected(hidden, N_CLASSES, activation="softmax")

# We optimize our model parameters using stochastic gradient descent.
sgd = tflearn.SGD(learning_rate=0.5)

# We can think of each layer of our networks as a separate model that performs a linear regression and whose output is passed through an activation function. Therefore, we will use TFLearn's deep neural network and regression functions to define our model. Furthermore, we do not need to create our own loss and accuracy operations -- TFLearn does it for us! 
network = tflearn.regression(y, optimizer=sgd, loss="categorical_crossentropy")
model = tflearn.DNN(network)

# Finally, we train our model, keeping track of how our loss and accuracy changes.
model.fit(Xtrain, Ytrain, n_epoch=EPOCHS, validation_set=(Xtest, Ytest), show_metric=True)

sess.close()

