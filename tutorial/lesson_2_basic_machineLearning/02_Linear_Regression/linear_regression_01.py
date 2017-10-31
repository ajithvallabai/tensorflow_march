'''
A gist of Linear regression using tensorflow 
Author:Ajith Kumar Veerapandian(Ajith Vallabai)

#linear_reg_hp.py and README_linear_hp.md
'''

import numpy
import matplotlib.pyplot as plt 
import tensorflow as tf

train_X=numpy.asarray([2104,1416,1534,852])
#train_X=area of the house in square feet
train_Y=numpy.asarray([460,232,315,178])
#train_Y=housing prices

m=train_X.shape[0]

X=tf.placeholder("float")
Y=tf.placeholder("float")

W=tf.Variable(0.25,name="weight")
b=tf.Variable(-0.02,name="bias")

hypothesis=tf.add(tf.multiply(X,W),b)

init=(tf.global_variables_initializer())
with tf.Session() as sess:
    sess.run(init)

    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()

