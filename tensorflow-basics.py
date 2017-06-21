import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
#from tensorflow.python.ops import rnn, rnn_cell



#------------------------------------------------------- MULTIPLICATION

#--------- Tensorflow is a symbolic library & uses symbolic expressions
#------------------------------ Creates symbolic variable of given type
a = tf.placeholder("float")
b = tf.placeholder("float")
#---------------------------------------- Multiplies symbolic variables
y = tf.multiply(a,b)
#--------------- tf.Session() creates a session to evaluate expressions
with tf.Session() as sess:
	#----------------------- Runs expressions (y) with given parameters
	print("%f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2}))
	print("%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3}))



#---------------------------------------------------- LINEAR REGRESSION

#------------------------ Creates evenly spaced numbers within interval
trX = np.linspace(-1,1,101)
#----------------------------- Creates y that's about linear with noise
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33
X = tf.placeholder("float")
Y = tf.placeholder("float")
#-------------------- Make linear regression by multiplying X by weight
def model(X,w):
	return tf.multiply(X,w)
#----------------------------- Create shared variable for weight matrix
w = tf.Variable(0.0, name="weights")
#-------------- To train model, use variables to hold/update parameters
#------------------- Variables are in-memory buffers containing tensors
y_model = model(X,w)
