#----------------------------------- Tensorflow is a symbolic library & uses symbolic expressions
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
#import rnn, rnn_cell



#--------------------------------------------------------------------------------- MULTIPLICATION

#-------------------------------------------------------- Creates symbolic variable of given type
a = tf.placeholder("float")
b = tf.placeholder("float")
#------------------------------------------------------------------ Multiplies symbolic variables
y = tf.multiply(a,b)
#----------------------------------------- tf.Session() creates a session to evaluate expressions
with tf.Session() as sess:
	#------------------------------------------------- Runs expressions (y) with given parameters
	print("%f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2}))
	print("%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3}))



#------------------------------------------------------------------------------ LINEAR REGRESSION
#------------------------------------- Models scalar dependent variable to EVs, infinite outcomes

#-------------------------------------------------- Creates evenly spaced numbers within interval
trX = np.linspace(-1,1,101)
#------------------------------------------------------- Creates y that's about linear with noise
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33
X = tf.placeholder("float")
Y = tf.placeholder("float")
#---------------------------------------------- Make linear regression by multiplying X by weight
def lin_model(X,w):
	return tf.multiply(X,w)
#------------------------------------------------------- Create shared variable for weight matrix
w = tf.Variable(0.0, name="weights")
#---------------------------------------- To train model, use variables to hold/update parameters
#--------------------------------------------- Variables are in-memory buffers containing tensors
y_model = lin_model(X,w)
#------------------------------------------------------------- Get square error for cost function
cost = tf.square(Y - y_model)
#----------------------------------- Constructs optimizer to minimize cost, then fit line to data
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#---------------------------------------------------------------------- Launch graph in a session
with tf.Session() as sess:
	#-------------------------------------------------- Runs initialization under Variable object
	tf.global_variables_initializer().run()
	for i in range(100):
		for (x,y) in zip(trX,trY):
			sess.run(train_op, feed_dict={X: x, Y: y})
	print sess.run(w)



#---------------------------------------------------------------------------- LOGISTIC REGRESSION
#------------------------------- Models categorical DV to EVs, outcome limited to possible values

#------------------------------------------- Init weights w/ shared variable, given shape, stddev
def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))
def log_model(X,w):
	#----------------------------------------------------- Same as linear reg; baked in cost f(x)
	return tf.matmul(X,w)
#--------------------------------------------------------- one_hot encodes data w/ oneHotencoding
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#---------------------------------------------------------- Assign train, test, and labels to var
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
#------------------------------------------------------ Create symbolic variables + weight matrix
X = tf.placeholder("float", [None,784])
Y = tf.placeholder("float", [None,10])
w = init_weights([784,10])
#-------------------------------------------------------------------- Model the logistic function
py_x = log_model(X,w)
#---------------------------------------------------------------- Get the mean of given parameter
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x,Y))
#---------------------------------------------------------------------------- Construct optimizer
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
#------------------------------------------------------------ Evaluate the argmax at predict time
predict_op = tf.argmax(py_x,1)
#---------------------------------------------------------------------- Launch graph in a session
with tf.Session() as sess:
	#------------------------------------------------------------------- Initialize all variables
	tf.initialize_all_variables().run()
	for i in range(100):
		for start,end in zip(range(0,len(trX),128), range(128,len(trX),128)):
			sess.run(train_op, feed_dict={X:trX[start:end], Y:trY[start:end]})
		print i, np.mean(np.argmax(teY,axis=1) == sess.run(predict_op,feed_dict={X:teX, Y: teY}))





