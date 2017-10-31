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
'''
#Dataset
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
                         
30.00,40.00,50.00,60.00,70.00,80.00,90.00,100.00,110.00,120.00,130.00,140.00,150.00
5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00                    
'''


m=train_X.shape[0]

X=tf.placeholder("float")
Y=tf.placeholder("float")

W=tf.Variable(0.18,name="weight")
b=tf.Variable(-40.0,name="bias")

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