
Let's study a little bit the available activation functions in Keras.  
Specifically, I'll do some studies for the following:
- sigmoid
- tanh
- softmax
- relu
- leaky relu
- elu

#### Classification problem


```python
import numpy as np

# Lets start by defining a fake dataset for a tipical classification problem with 3 classes
x = np.random.randn(5, 10)
y = np.array([[0,0,1], [0,0,1], [0,1,0], [0,1,0], [1,0,0]])

# we will have 5 inputs of 10 random parameters, and 5 outputs of 3 classes using one-hot encoding
print(x.shape, y.shape)
print(x[0])
print(y[0])
```

    (5, 10) (5, 3)
    [-0.6408213  -2.12716114 -0.83025388 -0.19434761  0.63255149  0.47409373
      0.68230783  0.93782778 -0.70996124  2.47312369]
    [0 0 1]


Now, we can just create a very simple shallow Keras model using those activations


```python
from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU
from keras.activations import sigmoid, tanh, softmax, elu, relu
```


```python
# sigmoid
model_sigmoid = Sequential()
model_sigmoid.add(Dense(3, input_shape=(10,), name="input"))
model_sigmoid.add(Activation(sigmoid, name="sigmoid"))

model_sigmoid.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (Dense)                (None, 3)                 33        
    _________________________________________________________________
    sigmoid (Activation)         (None, 3)                 0         
    =================================================================
    Total params: 33
    Trainable params: 33
    Non-trainable params: 0
    _________________________________________________________________



```python
# Let's first take a look at a few predictions of the model without training

predictions = model_sigmoid.predict(x)
print(predictions)
print("---")
print(predictions.max(), predictions.min())
print("---")
print(predictions.mean())
```

    [[0.948926   0.35752118 0.02653183]
     [0.4280658  0.54648334 0.6220165 ]
     [0.8289607  0.42473534 0.6102365 ]
     [0.60178727 0.7259306  0.3389298 ]
     [0.66332275 0.5656931  0.0648795 ]]
    ---
    0.948926 0.026531829
    ---
    0.5169347


As we can see, the random outputs of the untrained network seem to be sampled from the 0-1 interval.  
It has a mean of 0.5X (probably woud move towards 0.5 if we had much more data samples)


```python
import matplotlib.pyplot as plt
# Let's train the model now.
# Because the output is categorical, we will use categorical_crossentropy as the loss function, 
# and we will default to the simplest default Stochastic Gradient Descent as optimizer 
# to keep thigs as simple as possible

model_sigmoid.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
history = model_sigmoid.fit(x, y, epochs=1000, verbose=0)

plt.plot(history.history['loss'])
plt.legend(['loss'])
plt.show()

plt.legend(['accuracy'])
plt.plot(history.history['acc'])
plt.show()
```


![png](output_8_0.png)



![png](output_8_1.png)


As we can see, after 1000 epochs the model seems to only achieve a 80% accuracy (not so good for so many epochs!!!). The loss didn't seem to plateau so we can assume we could still keep training the model to achieve a better accuracy, but let's stop here, as we'll also train the other models with 100 epochs in order to keep the experiment as objective as possible.

And now, let's visualize the outputs of the trained network and see if something has changed. 


```python
# Let's now take a look at the same predictions after training the model

predictions = model_sigmoid.predict(x)
print(predictions)
print("---")
print(predictions.max(), predictions.min())
print("---")
print(predictions.mean())
```

    [[0.967454   0.06282943 0.7142852 ]
     [0.13549644 0.3292494  0.6389356 ]
     [0.60638165 0.8997735  0.6588457 ]
     [0.10091308 0.6893892  0.13126442]
     [0.7162901  0.16155303 0.27444643]]
    ---
    0.967454 0.06282943
    ---
    0.4724738


We still can see that the values are in the 0-1 interval, but surprisingly, after training (and with only 5 samples) the mean still seems to be around 0.5!!!

We can even see that some of the predictions have fairly high values like [0.6, 0.8, 0.6] so how based on this we could thing that the predictions are pretty bad :S

Let's process our utputs a little bit more to see why the accracy is 80%


```python
def force_prediction(preds):
    forced = []
    for p in preds:
        f = np.zeros((len(p)), dtype=np.int0)
        f[p.argmax()] = 1
        forced.append(f)
    return forced

predictions_processed = force_prediction(predictions)

print("expected:", y[2])
print("predicted:", predictions[2])
print("predicted (forced maz):", predictions_processed[2])
```

    expected: [0 1 0]
    predicted: [0.60638165 0.8997735  0.6588457 ]
    predicted (forced maz): [0 1 0]


As we can see, with this activation the model is learning to set the value at the expected position as the maximum value, but it's not very informative to have such high values, what does it mean?   
Does it means the model is not sure?  
It means the model is sure but that's how he learns?  

IMO it seems that **sigmoid** is NOT THE BEST activation for a classification problem

-----
Let's now do the same for **tanh**


```python
# tanh
model_tanh = Sequential()
model_tanh.add(Dense(3, input_shape=(10,), name="input"))
model_tanh.add(Activation(tanh, name="tanh"))

model_tanh.summary()

# Let's first take a look at a few predictions of the model without training
print("")
print("~~~ pre-trained predictions")
predictions = model_tanh.predict(x)
print(predictions)
print("---")
print(predictions.max(), predictions.min())
print("---")
print(predictions.mean())

# we now train the model
print("")
print("~~~ model training")
model_tanh.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
history = model_tanh.fit(x, y, epochs=1000, verbose=0)

plt.plot(history.history['loss'])
plt.legend(['loss'])
plt.show()

plt.legend(['accuracy'])
plt.plot(history.history['acc'])
plt.show()

# Now let's see the predictions after trainign
print("")
print("~~~ trained predictions")
predictions = model_tanh.predict(x)
print(predictions)
print("---")
print(predictions.max(), predictions.min())
print("---")
print(predictions.mean())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (Dense)                (None, 3)                 33        
    _________________________________________________________________
    tanh (Activation)            (None, 3)                 0         
    =================================================================
    Total params: 33
    Trainable params: 33
    Non-trainable params: 0
    _________________________________________________________________
    
    ~~~ pre-trained predictions
    [[-0.08869466 -0.83821374  0.98316574]
     [-0.45592272  0.58439714 -0.7200133 ]
     [ 0.68016255  0.01702786  0.9905297 ]
     [-0.38795567 -0.9820543   0.62676954]
     [-0.25264314 -0.80576444 -0.17423947]]
    ---
    0.9905297 -0.9820543
    ---
    -0.05489659
    
    ~~~ model training



![png](output_15_1.png)



![png](output_15_2.png)


    
    ~~~ trained predictions
    [[-0.9373205  -0.07852592  0.9965413 ]
     [-0.31441328  0.51666766 -0.7112085 ]
     [-0.9849429   0.96027046  0.98493123]
     [-0.12070431 -0.9847922   0.8359474 ]
     [-0.7937813  -0.45068717  0.4675483 ]]
    ---
    0.9965413 -0.9849429
    ---
    -0.04096464


Let's analyze this part by part.

About the pre-trained predictions, we can see an obvious range of -1 to 1, with a mean in 0. This is not very useful for us, as our output was composed of 0s and 1s so what would a -1 represent?

The training didn't go well at all, the model was not able to learn and the loss seems pretty chaotic :S

Finally, the predictions after training don't seem to show anything different from the ones before, this activation doesn't seem good at all for classification problems using one-hot encoddings :(

We can probably guess that the model could work better for a problem where the output goes from -1 to 1, but what kind of problem does need this? 

IMO it seems that tanh doesn't work for the outputs in classification problem. Would it be a good activation for a hidden layer? I don't know :S


----
Let's go now with **softmax**


```python
# softmax
model_softmax = Sequential()
model_softmax.add(Dense(3, input_shape=(10,), name="input"))
model_softmax.add(Activation(softmax, name="softmax"))

model_softmax.summary()

# Let's first take a look at a few predictions of the model without training
print("")
print("~~~ pre-trained predictions")
predictions = model_softmax.predict(x)
print(predictions)
print("---")
print(predictions.max(), predictions.min())
print("---")
print(predictions.mean())

# we now train the model
print("")
print("~~~ model training")
model_softmax.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
history = model_softmax.fit(x, y, epochs=1000, verbose=0)

plt.plot(history.history['loss'])
plt.legend(['loss'])
plt.show()

plt.legend(['accuracy'])
plt.plot(history.history['acc'])
plt.show()

# Now let's see the predictions after trainign
print("")
print("~~~ trained predictions")
predictions = model_softmax.predict(x)
print(predictions)
print("---")
print(predictions.max(), predictions.min())
print("---")
print(predictions.mean())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (Dense)                (None, 3)                 33        
    _________________________________________________________________
    tanh (Activation)            (None, 3)                 0         
    =================================================================
    Total params: 33
    Trainable params: 33
    Non-trainable params: 0
    _________________________________________________________________
    
    ~~~ pre-trained predictions
    [[0.16439128 0.01693138 0.8186773 ]
     [0.06454331 0.61529803 0.32015863]
     [0.43503037 0.30514285 0.25982675]
     [0.04082134 0.30678067 0.652398  ]
     [0.37682855 0.08465955 0.5385119 ]]
    ---
    0.8186773 0.016931383
    ---
    0.3333333
    
    ~~~ model training



![png](output_18_1.png)



![png](output_18_2.png)


    
    ~~~ trained predictions
    [[0.11211765 0.03024393 0.8576385 ]
     [0.0452106  0.11318032 0.84160906]
     [0.01408959 0.95173854 0.03417192]
     [0.03021203 0.91819483 0.05159311]
     [0.7878092  0.05215246 0.16003835]]
    ---
    0.95173854 0.014089586
    ---
    0.33333334


Let's analyze this part by part.

About the pre-trained predictions, we can see that the outputs seem to move in a range of 0-1 but, even though it might seem strange, there's a mean of 0,333. This is caused by the nature of the operation, the mean should be almost aways equal to 1 divided by the number of classes, so 1/3 = 0.333.  
Another interesting thing to note is that the values of each predictions always sum up to one (0.16+0.02+0.82 = 1) this gives us the hint that the output seems like a probability distribution.

About the training, after one partial success (sigmoid) and one failure (tanh) we're very suprised to see this model train so well! We can see that the accuracy reached the 100% after only ~180 epochs (we trained it for 1000) and the shape of the loss tells us that the model is capable to fit even more. Yay!

Finally, about the predictions after training, the mean and variance seem to be the same but looking at the values we can see that (because of the nature of the operation) the model has learned to power up some values close to one while keeping the rest close to 0, there is no ambiguety here like there was with sigmoid. It becomes very obvious to know what the model has learned, and what is the final prediction.

IMO it seems that sigmoid works VERY WELL as a final layer of a categorycal problem! 
Be careful as it will not work as well in hidden layers, as it squishes the hidden values too much, loosing to much information due to the fading gradient !!!

----
Let's go now with **relu**

This is the most popular and used activation, but it's generaly preferred as an activation in the hidden layers, let's see why


```python
# relu
model_relu = Sequential()
model_relu.add(Dense(3, input_shape=(10,), name="input"))
model_relu.add(Activation(relu, name="relu"))

model_relu.summary()

# Let's first take a look at a few predictions of the model without training
print("")
print("~~~ pre-trained predictions")
predictions = model_relu.predict(x)
print(predictions)
print("---")
print(predictions.max(), predictions.min())
print("---")
print(predictions.mean())

# we now train the model
print("")
print("~~~ model training")
model_relu.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
history = model_relu.fit(x, y, epochs=1000, verbose=0)

plt.plot(history.history['loss'])
plt.legend(['loss'])
plt.show()

plt.legend(['accuracy'])
plt.plot(history.history['acc'])
plt.show()

# Now let's see the predictions after trainign
print("")
print("~~~ trained predictions")
predictions = model_relu.predict(x)
print(predictions)
print("---")
print(predictions.max(), predictions.min())
print("---")
print(predictions.mean())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (Dense)                (None, 3)                 33        
    _________________________________________________________________
    relu (Activation)            (None, 3)                 0         
    =================================================================
    Total params: 33
    Trainable params: 33
    Non-trainable params: 0
    _________________________________________________________________
    
    ~~~ pre-trained predictions
    [[0.90719795 0.         2.2959151 ]
     [0.         0.33512932 0.        ]
     [0.         0.         0.50429004]
     [0.1005812  1.7293818  1.5515486 ]
     [0.         0.58397263 0.7209135 ]]
    ---
    2.2959151 0.0
    ---
    0.5819287
    
    ~~~ model training



![png](output_21_1.png)



![png](output_21_2.png)


    
    ~~~ trained predictions
    [[0.         0.         2.016315  ]
     [0.         0.45480382 0.        ]
     [0.         0.         0.8865733 ]
     [0.         2.3501985  0.        ]
     [0.         0.7701211  0.33278725]]
    ---
    2.3501985 0.0
    ---
    0.45405325


Let's analyze this part by part.

About the pre-trained predictions, the outputs are in a range of 0 to 2,3. Actually, relu doesn't impose a max for the range so the model is in the range of 0-infinity. The mean is 0.58, this doesn't seem to give any information by itself. Looking at the values we see some 0s and some positive values.

About the training, the loss seemes to have hit a plateau after around 200 epochs but the accuracy didn't seem to improve at all getting stuck at 0.4, not good, not goooood :(

Finally, about the predictions after training, the range seems to have increased a little bit, it's now 0 to 2.35 but the mean has decreased being now 0.45. This tells us that relu doesn't follow any kind of distribution, model learned to set some 0s and some positive values, but doesn't seem to have learned the correct predictions.

Why? Because of the nature of the operation, in a single layer of ReLU if the values are 0 there is no way the model will change that, so if the correct value is in a position that the initial prediction set to 0, it will never learn.
We can see a funny consecuence of this, and it's that if we randomize the values enough times so initially any of the correct values is predicted as 0, the model will work pretty well! But in a shallow model this doesn't work :(

IMO it seems that relu doesn't work well as an activation layer for the output of a classification problem. It COULD work though, if we remove the 0s problem by adding some hidden layers, but it will never be the ideal, because it could take much more time to learn if the outputs are very high.

----
Let's go now with **leaky relu**

This is variation of the ReLU that solves the 0s problem, so I'm going to guess that it will work better than the ReLU but let's see.


```python
# lrelu
model_lrelu = Sequential()
model_lrelu.add(Dense(3, input_shape=(10,), name="input"))
model_lrelu.add(LeakyReLU(0.01, name="relu"))

model_relu.summary()

# Let's first take a look at a few predictions of the model without training
print("")
print("~~~ pre-trained predictions")
predictions = model_lrelu.predict(x)
print(predictions)
print("---")
print(predictions.max(), predictions.min())
print("---")
print(predictions.mean())

# we now train the model
print("")
print("~~~ model training")
model_lrelu.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
history = model_lrelu.fit(x, y, epochs=1000, verbose=0)

plt.plot(history.history['loss'])
plt.legend(['loss'])
plt.show()

plt.legend(['accuracy'])
plt.plot(history.history['acc'])
plt.show()

# Now let's see the predictions after trainign
print("")
print("~~~ trained predictions")
predictions = model_lrelu.predict(x)
print(predictions)
print("---")
print(predictions.max(), predictions.min())
print("---")
print(predictions.mean())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (Dense)                (None, 3)                 33        
    _________________________________________________________________
    relu (Activation)            (None, 3)                 0         
    =================================================================
    Total params: 33
    Trainable params: 33
    Non-trainable params: 0
    _________________________________________________________________
    
    ~~~ pre-trained predictions
    [[ 9.3719065e-01  1.9308243e+00 -2.5536511e-02]
     [-2.9509276e-04  6.1618912e-01  5.6603360e-01]
     [-9.8429620e-03  8.8597006e-01  1.6449414e+00]
     [ 2.6999810e+00  5.3981167e-01 -1.0586169e-02]
     [ 1.3728960e+00  4.5086497e-01 -1.8345760e-02]]
    ---
    2.699981 -0.025536511
    ---
    0.7720064
    
    ~~~ model training



![png](output_24_1.png)



![png](output_24_2.png)


    
    ~~~ trained predictions
    [[ 5.0316197e-01  1.7596383e+00 -3.2398224e-02]
     [-5.9180479e-03  4.2344630e-04  1.0608001e+00]
     [-3.3544479e-03  1.1593986e+00 -2.0972582e-05]
     [ 5.5771619e-03  2.0479093e+00 -5.7644579e-03]
     [ 8.1088138e-01  6.7752451e-03 -2.0072155e-02]]
    ---
    2.0479093 -0.032398224
    ---
    0.4858025


Let's analyze this part by part.

About the pre-trained predictions, the outputs are in a range of -0.03 to 2,7. Actually, relu doesn't impose a max nor a min for the range so the model is in the range of -infinity to infinity. 
We can see though that the movement towards negative values is much smaller, this is caused by the nature of the operation, that multiplies the negative values by x0.01 (this is a parameter, I choose this one arbitrarily).
The mean is 0.77, but as with ReLU this doesn't seem to give any information by itself.

About the training, the loss seemes to decrease properly, but sadly it hits a plateau after 380 epochs. At around 200 epochs the model hits its max accuracy of 80%

Finally, about the predictions after training, the range seems to have shifter a little bit towards 0, it's now 0.033 to 2 but the mean has decreased being now 0.48. This tells us that the leaky relu also doesn't follow any kind of distribution.

IMO it seems that leaky relu is also NOT THE BEST as an activation layer for the output of a classification problem. 
It solves the problem with the 0s that had the ReLU model so it works much better, but it still has a problem fitting high or low values to 0 and 1.

----
Let's go now with **leaky elu**

Finally lets try elu. It's a mistery for me as I do not know the implementation of it, but hopefully this analysis will help us undersdand it better.


```python
# lrelu
model_elu = Sequential()
model_elu.add(Dense(3, input_shape=(10,), name="input"))
model_elu.add(Activation(elu, name="elu"))

model_elu.summary()

# Let's first take a look at a few predictions of the model without training
print("")
print("~~~ pre-trained predictions")
predictions = model_elu.predict(x)
print(predictions)
print("---")
print(predictions.max(), predictions.min())
print("---")
print(predictions.mean())

# we now train the model
print("")
print("~~~ model training")
model_elu.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
history = model_elu.fit(x, y, epochs=1000, verbose=0)

plt.plot(history.history['loss'])
plt.legend(['loss'])
plt.show()

plt.legend(['accuracy'])
plt.plot(history.history['acc'])
plt.show()

# Now let's see the predictions after trainign
print("")
print("~~~ trained predictions")
predictions = model_elu.predict(x)
print(predictions)
print("---")
print(predictions.max(), predictions.min())
print("---")
print(predictions.mean())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (Dense)                (None, 3)                 33        
    _________________________________________________________________
    elu (Activation)             (None, 3)                 0         
    =================================================================
    Total params: 33
    Trainable params: 33
    Non-trainable params: 0
    _________________________________________________________________
    
    ~~~ pre-trained predictions
    [[-0.9232144   1.8610909   0.9773937 ]
     [ 0.9129013  -0.2885369  -0.44073778]
     [ 1.5642755   1.8806123   1.7354904 ]
     [-0.7151167   0.4838999  -0.07944727]
     [-0.81829035  0.5162267   0.7964347 ]]
    ---
    1.8806123 -0.9232144
    ---
    0.49753216
    
    ~~~ model training



![png](output_27_1.png)



![png](output_27_2.png)


    
    ~~~ trained predictions
    [[ -0.94915956  -1.           1.2272292 ]
     [  1.0288106   -1.          -0.39465106]
     [  0.33675778 251.8037       0.61315924]
     [ -0.5990375   -1.           0.6220208 ]
     [ -0.83801085  -1.           1.0730296 ]]
    ---
    251.8037 -1.0
    ---
    16.661589


Let's analyze this part by part.

About the pre-trained predictions, the outputs are in a range of -0.93 to 1.9, the mean of 0.5 could give us some hint about the distribution as it's (max+min)/2 , though it could be just a coincidence, or could be a reflection of the intial random weight distribution if the elu operation doesn't reflect a distribution by itself.

About the training, it's a total disaster, not much to say that the graphs don't tell... :(

Finally, about the predictions after training, the range seems to have changed a lot, from -1 to 252!!! 
Looking at the data, there seems to be a minimum at -1 as a lot of values have taken that value and doesn't seem accidental, but it's really weird that a single value has increased to 251 while the rest seem to be kept pretty low!! I don't know the reason for that :( 

About the mean, it's become clear that the previous assumption was just a coincidence, and the mean doesn't seem to follow any distribution.

IMO, as the other ReLU and variations, this doesn't seem like a good activation for classification. In previous experiments I've seen a good performance of this layer for regression problems, but I'm not sure why. It seems to keep some kind of smooth relation between the data like Sigmoid, keeps some properties of ReLU by keeping prositive values unaltered and avoids the 0s problem of leakyRely by setting the min to -1.
