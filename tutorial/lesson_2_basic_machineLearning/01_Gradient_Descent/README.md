###Objective is to minimize the loss(log(x)^2) and find the coressponding x value

```
import tensorflow as tf

x = tf.Variable(4, name='x', dtype=tf.float32)
```
x is made as variable so that the function (tf.train.GradientDescentOptimizer().minimize(loss))
could vary x to minimize loss
variable x is declared with 4 as it value and datatype as float32 so x=4.0 
Note utill session is ran x is not assigned

`log_x = tf.log(x)`
log_x becomes  0.69314=log (2)
`log_x_squared = tf.square(log_x)`
log_x_squared = 0.48045 = log_x^2

here 0.3 is the learning rate it could changed  with in considerable 
limits which makes the gd to move and minimize function will make 

`train=tf.train.GradientDescentOptimizer(0.3).minimize(log_x_squared)`

Also can be declared as
optimizer = tf.train.GradientDescentOptimizer(0.3)
train = optimizer.minimize(log_x_squared)
i need to write explanation for tf.train.GradientDescentOptimizer().minimize() function


`init = tf.initialize_all_variables()`
initializing all variables so that it could be used
but until session.run(init) is ran its not done

the below method is one of the method how session is initilzed in  tensorflow
```
with tf.Session() as session:
  session.run(init)
  #session is initialised here officially
  print("starting at", "x:", session.run(x), "log(x)^2:", session.run(log_x_squared))
  #shows the initial values of x and log_x_squared

  #range is set to 10 -->means tf.train.Gd.minimize() is going to be used for minimizing
  #the loss(log(x)^2) 
  #note the function(tf.train.gd.minimize) is programmed to move every step towards minimal loss
  #provided the learing rate is small

  for step in range(10):  
    session.run(train)
    print("step", step, "x:", session.run(x), "log(x)^2:", session.run(log_x_squared))
```

Story:
The story moves in such a way that it compares real life scenario with the (program)
Situation:Imagine if vallabai in middle of a mountain and if you want to move towards the foothill
Vallabai has two legs and he is tired .the steps that vallabai should take will have a maximum limit(here that is 0.3)
and the destination(direction) is his home(in this problem the destination log(x)^2 to minimum(0) ) at foothill(distance to destination=0)
Start = Point A
distance to destination = log(x^2)
Home = Point Z

Now from middle of the mountian(point A) vallabai needs to reach home with tired legs so he is waking with
one objective that is you should take minimum number of steps with 0.3 step effort and reduce the distace(logx^2)
to reach home where(logx^2=0)

Explanation:
Read carefully here from the point A vallabai thinks(gradientoptmizer function is the brain signal (inside gradient optimizer.
it thinks(value of x is chosen within gradient optimizer formula(need to see how it works)) to move up(x=4.2) ,right(imaginative) 
or left(imaginative) or down(x=3.79206) the sole objective is to reach home so he takes the downward path towards foothill)) brain
sees his home downwards(where log(x^2 is minimum reduced to 1.77664 from 1.92181))
so he takes his first step with 0.2079 (4.0-3.79206 = 0.2079) in downward direction to point B.From point B
he thinks for his next step (gradientoptimizer function works again) and as the home is downward direction 
his brain(gradient optimizer function ) tells to move downwards and he takes further steps in similar manner
and reaches Point Z "Vallabai's home".


*Result*
Starting at x: 4.0 log(x)^2: 1.92181    | *Note the value of x starts at 4 with log(4^2) value = 1.92181*
    step 0 x: 3.79206 log(x)^2: 1.77664 |  By the function gradientdescentoptimizer with 0.3 as its stepping 
                                        |  limit value takes a step of(4.0-3.79206 = 0.2079) 0.2079 and 
                                        |  thus minimising log(x^2) to 1.77664 .
    step 1 x: 3.58116 log(x)^2: 1.62737 |  Similarly with 0.3 as the stepping limit takes another 
                                        |  step of 0.2109 (3.79206-3.58116 =0.2109) and minimising the log(x^2) value to 
    step 2 x: 3.36742 log(x)^2: 1.47415 |
    step 3 x: 3.15109 log(x)^2: 1.31733 |
    step 4 x: 2.93255 log(x)^2: 1.1575
    step 5 x: 2.71242 log(x)^2: 0.995689
    step 6 x: 2.49169 log(x)^2: 0.833502
    step 7 x: 2.27185 log(x)^2: 0.673378
    step 8 x: 2.05513 log(x)^2: 0.51889
    step 9 x: 1.84483 log(x)^2: 0.375017
    step 10 x: 1.64566 log(x)^2: 0.248145
    step 11 x: 1.46404 log(x)^2: 0.145313
    step 12 x: 1.30781 log(x)^2: 0.0720157
    step 13 x: 1.1847 log(x)^2: 0.0287259
    step 14 x: 1.09886 log(x)^2: 0.00888727
    step 15 x: 1.04738 log(x)^2: 0.00214331
    step 16 x: 1.02086 log(x)^2: 0.000426371
    step 17 x: 1.00873 log(x)^2: 7.55065e-05
    step 18 x: 1.00356 log(x)^2: 1.26198e-05
    step 19 x: 1.00143 log(x)^2: 2.05571e-06
    step 20 x: 1.00058 log(x)^2: 3.31333e-07
    step 21 x: 1.00023 log(x)^2: 5.31964e-08
    step 22 x: 1.00009 log(x)^2: 8.5126e-09
    step 23 x: 1.00004 log(x)^2: 1.36561e-09
    step 24 x: 1.00001 log(x)^2: 2.18503e-10
    step 25 x: 1.00001 log(x)^2: 3.55269e-11
    step 26 x: 1.0 log(x)^2: 5.68433e-12
    step 27 x: 1.0 log(x)^2: 9.09494e-13
    step 28 x: 1.0 log(x)^2: 1.27898e-13
    step 29 x: 1.0 log(x)^2: 1.42109e-14
    step 30 x: 1.0 log(x)^2: 0.0
    step 31 x: 1.0 log(x)^2: 0.0
    step 32 x: 1.0 log(x)^2: 0.0
    step 33 x: 1.0 log(x)^2: 0.0
    step 34 x: 1.0 log(x)^2: 0.0
    step 35 x: 1.0 log(x)^2: 0.0
    step 36 x: 1.0 log(x)^2: 0.0
    step 37 x: 1.0 log(x)^2: 0.0
    step 38 x: 1.0 log(x)^2: 0.0
    step 39 x: 1.0 log(x)^2: 0.0
    step 40 x: 1.0 log(x)^2: 0.0
    step 41 x: 1.0 log(x)^2: 0.0
    step 42 x: 1.0 log(x)^2: 0.0
    step 43 x: 1.0 log(x)^2: 0.0
    step 44 x: 1.0 log(x)^2: 0.0
    step 45 x: 1.0 log(x)^2: 0.0
    step 46 x: 1.0 log(x)^2: 0.0
    step 47 x: 1.0 log(x)^2: 0.0
    step 48 x: 1.0 log(x)^2: 0.0
    step 49 x: 1.0 log(x)^2: 0.0


 

from above you could see that log(x)^2 is decresing at every step
Thus our our objective is obtained log(x)^2=0.0(minimal loss) with  x=1.0



Gradeintdescentent fucntion (tf.train.GradientDescentOptimizer().minimize()) -explanation
Automatic differentiation -- https://en.wikipedia.org/wiki/Automatic_differentiation
This is one of the main reason for tensorflow existense it could calcualte with internal graph
and detect,combine all those functions and variables with in the graph and provide
the desired output
Tensorflow always track the path of operations. I mean the sequential behavior of the nodes and 
 how data flow between them. That is done by the graph
https://stats.stackexchange.com/questions/257746/how-does-tensorflow-tf-train-optimizer-compute-gradients

(need to explain tf.train.GradientDescentOptimizer())