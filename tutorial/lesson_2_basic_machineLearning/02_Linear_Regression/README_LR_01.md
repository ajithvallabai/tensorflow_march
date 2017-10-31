Storyline:Vallabai wants to get a home for considerable price
#linear_reg_hp.py and README_linear_hp.md
```
Scientific artistry-
Artistic explanation for science
```

Vallabai has relocated to a new place called bangalore('Silicon valley of east').it has been 1 year since he has been paying lumpsome rent hence after doing lot of discussions with his mind he decides that it would be better if he gets a home in bangalore.
(Other parameters are not included as it is a simple starter program)

Vallabai thinks to restrict the brokarage fee by beliving in his data analysis skill

he had putforth this topic during the lunch time to few off his friends
they told their prices on which they got
(Other parameters are not included as it is a simple starter program kindly refer further programs for additional parameters)
#krishna
ram told that he got a 2104 sqft house for 460k 
shiva told that he got a 1416 sqft house for 232k 
nera told that he got a 1534 sqft house for 315k 
deepak told that he got a 852 sqft house for 178k

vallabai got his dataset now(no.of.datas set=4) 
```
train_X=numpy.asarray([2104,1416,1534,852])
#train_X=area of the house in square feet
train_Y=numpy.asarray([460,232,315,178])
#train_Y=housing prices

m=train_X.shape[0]

X=tf.placeholder("float")()
Y=tf.placeholder("float")
````

Vallabai's had a device named dator which could give him a hypothesis(a equation) that could predict the housing prices with the given data set note with the
given data set right.

after feeding data to dator
the device showed that 

'hypothesis=(-40.0+0.25x)'

```
W=tf.Variable(0.25,name="weight")
b=tf.Variable(-40.0,name="bias")

hypothesis=tf.add(tf.multiply(X,W),b)
```
(framing the hypothesis is a important task for linear regression
will explain in further programs on how to frame it )


Vallabai loves machines and the device bluprint showed that it was designed by ajith so he trusted the equation and ajith

(Note formula and hypothesis are different.formula is a proof but 
hypothesis is a prediction

for example formula: area of a rectangle=length of breadth * height
whereas hypothesis : As the precipitation(0.6) and humidity(0.9) is high today it
may rain --> precipitation rate+ hymidity rate/2 = (0.6+0.9)/2 = 0.75 
so 75% it will rain today

hypothesis may be 99% right but formula should be 100% right wrt conditions
)

Now vallabai is doing the calculations with the given dataset 
sqft(X) -price(OP)     hypothesis(-40.0+0.25x)  resulting price(predicted)
2104    -460        -40.0+(0.25*(2104))         486
1416    -232        -40.0+(0.25*(1416))         314
1534    -315        -40.0+(0.25*(1534))         344
852     -178        -40.0+(0.25*(852))          173


Graphs:Are very important concept in science and mathematics to know where 
we are in the problem (for example  when you are lost while searching for some shop or a friends home ,maps(graphs) provide you getting a direction and your location in google maps).ya graphs will help you to visuvalise 
and plot your problem clearly in a consile manner

Vallabai takes a graph and plots the X and original price told by his friends

(attach graph 1)

(these are original plots)

then he takes the resulting price(predicted price and maps in another graph)

(attach graph 2)
(these are predicted plots)

When you join the dots you get a line that is what we want - regression line
(the line is drawn on the hypothesis equation(derived from feeding original sqft plots to ajith's device))
if we get that line then you could predict the price range for which ever 
house you want

when you sysnchronise both the graphs
you get
(attach graph 3)

so now with tht line we could check how much does 3000 squareft house costs
in bangalore answer-
(attach graph 4 - draw and connect the 3000 sqft with y axis)

Thus the magic ends

-----------------------------------------
"An oops is better than what if" 


what if vallabai has had three other devices and each of the
device gave him different hypothesis 

device 1 --> hypothesis-1 = 
device 2 ---> hypothesis-2 =
device 3 ---> hypothesis-3 =

vallabai saw a quote in his near by desk "An oops is better than what if" so vallabai goes in accordance with it because he likes it

he tries each and every hypothesis

hypothesis 2 =

hypothesis 3 =


Dont underestimate the time spent for hypothesis 2 and hypothesis 3 
only bad hypo could help you differenitate the
good hypo
i need to write further



