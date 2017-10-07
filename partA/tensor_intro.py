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

session_launch.close()


a=tf.constant(20,name="a")
b=tf.constant(30,name="b")

x=tf.add(a,b,name="addition")
y=tf.add(a,b,name="division")

#method one to use sess
sess=tf.Session()

result_one=sess.run(x)
result_two=sess.run(y)
print("The result of addition is",result_one)
print("The result of addition is",result_two)
sess.close()


#method two to use sess
sess=tf.Session()
with tf.Session() as sess:
    print("The answer of addition is",sess.run(x))
    print("The answer of addition is",sess.run(y))

sess.close()
