#final-->logistic regression-->single variable
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 

#train_names=(1,2,3,4,5)
#train_X=np.array([10,20,30,40,50,60,70,80,90,100,110,120])
#train_X=np.array([-6,-5,-4,-3,-2,-1,1,2,3,4,5,6])
#took the data set from wikipedia logisitc regression explanation

#train_X -->denotes the no.of hours a person is working
train_X=np.array([0.50,0.75,1.00,1.25,1.50,1.75,1.75,1.85,2.00,2.25,2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.75,5.00,5.50])

#train_X_var=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
#tested whether the train_X_var could be used as a x but its index so dont be stupid to use it

#two classs are there 0 and 1
#train_Y=np.asmatrix([0,0,0,0,0,0,1,1,1,1,1,1]).T

#train_Y-->denotes that whether a person is able to complete a task or not
train_Y=np.asmatrix([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]).T

train_X=np.asmatrix(train_X).T

#number of X and Y value
m=train_X.shape[1]

#learning_rate
learning_rate = tf.constant(0.1,dtype=tf.float32)
training_cycle = 10000

#X--> train_X ,Y-->train_Y
#w --> weight ,b--> bias
#remember e^(rt+c) --> t(time) is the dependent variable so it is modified as x which is the dependent variable in this problem  
#w is the effort or weight --> in (e^rt+c) --> r is the rate of change and c is the account of losses or gain during those calculations like accounting the other enviromental facotrs
#same way w is the rate at which the variables are changing with respect to x and the accounting of loss and gain in calculations is done with the constants
#w will matter most at some places and b will matter at some places but only because of w ,b is existing 
#these loss and gains are different from the below loss and gain
X = tf.placeholder(tf.float32, [train_X.shape[0],m])
y = tf.placeholder(tf.float32, [train_Y.shape[0],1])
w = tf.Variable(np.ones([m,1]),dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

#initialization of all variables
init = tf.initialize_all_variables()

#sigmoid function..contains the constant fo nature
hypothesis = (1./(1+tf.exp(tf.matmul(X,w) + b)))

#loss 
loss = tf.reduce_mean(tf.square(hypothesis - y))
#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


#initialize session
sess = tf.Session()
sess.run(init)

loss_history = []
for epoch in range(training_cycle):
    sess.run(optimizer,feed_dict={X: train_X, y: train_Y})
    loss_history.append(sess.run(loss,feed_dict={X: train_X, y: train_Y}))
    if(epoch%1000==0):
        print(sess.run(w))
        print(sess.run(b))

print("finished optimizing")

print("Showing original data")
print("Are you able to see the two classes")
plt.plot(train_X,train_Y,'ro',label='Original data')
plt.show()

print("The fitted line")
print("Did you see the fitted line is a pure sigma fucntion curve cool right it shows the growth in the perios of birth and death of a entity its too deep")
print("want to know why 1/(1+e^),https://www.khanacademy.org/math/ap-calculus-bc/bc-diff-equations/bc-logistic-models/v/solving-logistic-differential-equation-part-1")
print("I too have my explanations in detail will publish later")

plt.plot(train_X,1./(1+np.exp(np.dot(train_X,sess.run(w)) + sess.run(b))),label='Fitted_line')
plt.legend()
plt.show()



print("Printing the loss")
plt.plot(range(len(loss_history)),loss_history)
plt.show()
print("I dont know how the 1/x curve is made for loss function for this")


w = sess.run(w)
b = sess.run(b)
print("W: %.4f" % w) 
print("b: %.4f" % b) 

print("The current probablity of classifying the value is here")
print (1./(1+np.exp(np.dot(train_X,(w)) + (b))))

