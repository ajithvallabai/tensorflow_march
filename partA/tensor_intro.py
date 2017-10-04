import tensorflow as tf

#declaring contants and calculating the result
matrixA=tf.constant([[1.,2.,3.]])
matrixB=tf.constant([[1.],[2.],[3.]])
product=tf.matmul(matrixA,matrixB)

#Launching graph in session
session_launch=tf.Session()

#executing of operation with session
answer=session_launch.run(product)
print("The calculated value is ",answer)
session_launch.close()

#creating tensor with random values
randomValues=tf.random_normal([4,5])
session_launch=tf.Session()
session_launch.run(randomValues)
