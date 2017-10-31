dator:Device configuration

last story line:

vallabai was impressed with the dator device he got a
new house.Now his mind desires to search how was the device
configured so he is reverse engineering the device .Now vallabai
is hungry so he went to prepare his meal,then he ate now when he
started to know about device in deep he was not able to
understand it so he started to sleep.Great right enigneers work 
like that.Now vallabai is in Rapid eye movement(REM) sleep he is dreaming come lets go into his dream.

Now Vallabai is getting down from the mountain after touching the peak he is too tired as it was a long journey.He is in the middle of the mountain surrounded by evergreen forest cool right but not for him he doesnt have a smart phone hahaha.Smart people doesnt require a smart phone. skills of vallabai are so advanced and he is a good tree climber .Almost he is lost but due to his tree climbling skills(learning_rate) he was able to see his home.
`learning_rate=0.000001`

He would be approximately 1000 steps away from his home.
`training_cycle=1000`

<-- fill in he is watching the direction of his home by climbing
an dnoting the learning_rate and the form there he go down and then walks futher
towards his home and with minimum steps and reach the home
-->



```
hypothesis=tf.add(tf.multiply(X,W),b)
loss=tf.reduce_sum(tf.pow(hypothesis-Y,2))/(2*m)
learning_rate=0.000001
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
training_cycle=1000


init=(tf.global_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    #print("W=", sess.run(W), "b=", sess.run(b), '\n')
    for cycle_no in range(training_cycle):
        #print("W=", sess.run(W), "b=", sess.run(b), '\n')
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})

        #print("Epoch %d,loss=%f %(cycle_no,loss)")
    print("finished optimizing")
```

