#objective is to print all values from 0 to 5 with whileloop
import tensorflow as tf 

x=tf.Variable(0.,name='x')
thereshold=tf.constant(5.)

init=tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    while session.run(tf.less(x,thereshold)):
        x=x+1
        result=session.run(x)
        print(result)
