###objective to print values from 4 to 8 with forloop

'''
import tensorflow as tf 
a_value=tf.Variable(3,name='a_value')
'''
declaring a_value=3 as a tf.Variable()

'''
init=tf.global_variables_initializer()
'''
initilazing gloabl_variable_initilizer()


*Note only when tf.Session.run(init) or tf.Session.run(a) is called "init" or "a"  declaration is valid(executed)*

'with tf.Session() as session:'

declaring session


    'for i in range(5):'
        'session.run(init)'
        *also can be used as  tf.Session.run(init)*

        'a_value=a_value+1'
        *incrementing value*

        'print(session.run(a_value))'
        *printing*
