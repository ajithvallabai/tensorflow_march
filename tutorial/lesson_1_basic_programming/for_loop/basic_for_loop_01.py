#objective to print values from 4 to 8 with forloop

import tensorflow as tf 
a_value=tf.Variable(3,name='a_value')
init=tf.global_variables_initializer()
with tf.Session() as session:
    for i in range(5):
        session.run(init)
        a_value=a_value+1
        print(session.run(a_value))
