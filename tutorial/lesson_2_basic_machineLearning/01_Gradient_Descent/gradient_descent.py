#Objective is to minimize the loss(log(x)^2) and find the coressponding x value
import tensorflow as tf
x = tf.Variable(4, name='x', dtype=tf.float32)
log_x = tf.log(x)

log_x_squared = tf.square(log_x)
train=tf.train.GradientDescentOptimizer(0.3).minimize(log_x_squared)
init = tf.initialize_all_variables()
with tf.Session() as session:
  session.run(init)
  print("starting at", "x:", session.run(x), "log(x)^2:", session.run(log_x_squared))
  for step in range(50):  
    session.run(train)
    print("step", step, "x:", session.run(x), "log(x)^2:", session.run(log_x_squared))

