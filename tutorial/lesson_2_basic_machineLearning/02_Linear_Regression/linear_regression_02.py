'''
A gist of Linear regression using tensorflow 
Author:Ajith Kumar Veerapandian(Ajith Vallabai)

#linear_reg_hp.py and README_linear_hp.md
'''

import numpy
import matplotlib.pyplot as plt 
import tensorflow as tf

train_X=numpy.asarray([2104.0,1416.0,1534.0,852.0])
#train_X=area of the house in square feet
train_Y=numpy.asarray([460.0,232.0,315.0,178.0])
#train_Y=housing prices




m=train_X.shape[0]

X=tf.placeholder("float")
Y=tf.placeholder("float")

W=tf.Variable(0.0,name="weight")
b=tf.Variable(0.0,name="bias")

hypothesis=tf.add(tf.multiply(X,W),b)

loss=tf.reduce_sum(tf.pow(hypothesis-Y,2))/(2*m)

learning_rate=0.000001

training_cycle=1000


optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


init=(tf.global_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    #print("W=", sess.run(W), "b=", sess.run(b), '\n')
    for cycle_no in range(training_cycle):
        #print("W=", sess.run(W), "b=", sess.run(b), '\n')
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})

        #print("Epoch %d,loss=%f %(cycle_no,loss)")
    print("finished optimizing")

    training_cost = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')


    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()
#524.406 -- 0.0000001 -- 2000
#524.405 --- 3000
#524.308 --- 3000 - 0.00000001

    test_X = numpy.asarray([900.0, 2000.0, 1500.0])
    test_Y = numpy.asarray([185.0, 440.0, 250.0])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()