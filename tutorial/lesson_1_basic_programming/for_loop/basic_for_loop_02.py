import tensorflow as tf 
#importing tensorflow

a=tf.Variable(5)
#declaring tensor variable a=5

init=tf.global_variables_initializer()
#initializing all tensor global variables

#Note only when tf.Session.run(init) or tf.Session.run(a) is called "init" or "a"  declaration is valid(executed) 

session=tf.Session()
#declaring session

for i in range(5):
    session.run(init)
    #also can be declared as  tf.Session.run(init)
    a=a+1
    print(session.run(a))
    #also can be used as  print(tf.Session.run(a))