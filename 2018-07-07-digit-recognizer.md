---
layout: post
title: "Kaggle: Digit Recognition"
date: 2018-07-07T10:20:00Z
categories: Kaggle
---
<br>
So this blog post is actully going to be a little different because this has nothing to do with GSOC(Google Summer of Code) but well it could because some of the projects have to deal with NLP(Natural Language Processing) but for now we will be dealing with a competition relating to digit recognition. So essentially we want to teach the computer to be able to read hand written digits. The images generated are a 28 * 28 image that is in grey scale. Now working on the project was myself Keanu Nichols and [Akil Hosang]() we used a Neural Network approach and we tried to model it after the Coursera Machine Learning (ML) course by Andrew Ng. We did a very simple implentation by the way, it was for us to simply work on a problem since we were working on the ML course and wanted to get some actual experience working on a project so we looked to [Kaggle]() and one of the beginner projects was this. Overtime I hope we can improve the score of this approach which I believe is possible. So let's dive into it.


First we go about importing our different modules as seen below, pandas is used for dataframes to store the pixels of the images and do a number of different operations. Matplotlib is used to actually see the image using the dataframes storing the pixels. Numpy is used to manipulate array to do things like transposing a matrix or multiplying two matrices. Random is used to give random numbers as will be seen in the function "randInit". And finally minimize which is an equivalent to fmincg in matlab which is used to train the actual Neural Network.

{% highlight python linenos %}
# Plot ad hoc mnist instances
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import random
import math
from scipy.optimize import minimize

{% endhighlight %}

Then we go onto loading our data into a training set 

{% highlight python linenos %}
# load (downloaded if needed) the Kaggle dataset
train = pd.read_csv('./datasets/train.csv')
train.head()
{% endhighlight %}

{% include digit_rec_train.md %}

and test set. 

{% highlight python linenos %}
test = pd.read_csv('./datasets/test.csv')
test.head()
{% endhighlight %}


{% include digit_rec_test.md %}



Since we would want to just have the pixels we take the first column and onwards to get all the pixel data and store it in X_train. We then store the labels of the numbers for e.g. if the number is 2 or 3 and we store that in y_train. After we train our test date into X_test, we will actually feed this to the Neural Nwtwork and get our outputs and submit to Kaggle.

{% highlight python linenos %}
X_train = (train.iloc[:,1:].values.astype('float32'))
y_train = (train.iloc[:,0].values.astype('int32'))
X_test = test.values.astype('float32')
{% endhighlight %}

Now the order in which I do this will not be the exact formate of the jupyter notebook but this is for us to better understand the order in which it is executed. So the first part is to store our data into the Neural Network class.

{% highlight python linenos %}
nn = Neural_Network(X_train,np.transpose(y_train))
{% endhighlight %}

The Neural_Network class looks like this and we see that it's initialized with m the number of inputs which in our case will be 784 (28 * 28) our input layer is set to 784, hidden layer is 25 (was used in the exercise for the ML course) and our output layer size is 10 (because we have 10 digits 0-9). Epsilon is what we're going to calculate soon enough. J is our cost function and grad is our graident. Num_labels I think is self explanitory the number of labels and lambda1 is lambda which will be used in both forward and back propagation, iter is used when we are going through using minimize for us to see our iteration number of training the Neural Network.
{% highlight python linenos %}
class Neural_Network:
    #The Neural_Network class we use to store all the values that
    #will be used to teach our model to start recognizing handwritten
    #digits
    def __init__(self,X,y):
        self.m = X_train.shape[0]
        self.input_layer_size = X_train.shape[1]
        self.hidden_layer_size = 25
        self.output_layer_size = 10
        self.epsilon = []
        self.J = 0
        self.grad = 0
        self.num_labels = 10
        self.lambda1 = 0
        self.iter = 0
{% endhighlight %}

So we go forward actually getting theta1 (for our pixels) and theta2 (for our number labels). We wil call the function "randInit()" to actually populate it with values.

{% highlight python linenos %}
theta1 = theta2 = []
theta1,theta2 = nn.randInit()
{% endhighlight %}

We first go out getting our values for epsilon, which was given by the [ML course]() (under Random Initialization), we split it for theta1 and theta2 that's why epsilon is a list of two values. We also add our bias unit. We then store our values for theta1 and theta2 by random initalization. It is more or less using what's in the course.
{% highlight python linenos %}
    def randInit(self):
        theta1 = []
        theta2 = []
        for x in range(2):
            if(x == 0):
                L_in = self.input_layer_size
                L_out = self.hidden_layer_size
            if(x == 1):
                L_in = self.hidden_layer_size
                L_out = self.output_layer_size
            self.epsilon.append(math.sqrt(6) / math.sqrt(L_in + L_out))
            
        X_bias = self.input_layer_size + 1
        
        theta1 = (np.random.rand(self.hidden_layer_size,
                 self.input_layer_size + 1) * (2 * self.epsilon[0])
                 - self.epsilon[0])
        theta2 = (np.random.rand(self.output_layer_size,
                  self.hidden_layer_size + 1) * (2 * self.epsilon[1])
                  - self.epsilon[1]) 
        return theta1,theta2

{% endhighlight %}




Resources:
GSoC ideas (Specifically Ideas 2 & 3): [Ideas](https://wiki.linuxfoundation.org/chaoss/gsoc-ideas)<br>
My proposal: [My proposal](https://github.com/kmn5409/chaoss-microtasks/blob/master/GSoC-2018-Keanu-Nichols-CHAOSS-proposal.pdf)


Files Used:
Python File - [PiperRead 12](https://github.com/kmn5409/GSoC_CHAOSS/blob/master/Augur/Perceval/PiperReader%2012.py#L149)<br>
Jupyter Notebook - [PiperMail 8](https://github.com/kmn5409/GSoC_CHAOSS/blob/master/Augur/Perceval/PiperMail%208.ipynb)

Jupyter Notebook - [Sentiment_Piper 6](https://github.com/kmn5409/GSoC_CHAOSS/blob/master/Augur/Perceval/NLP/Sentiment_Piper%206.ipynb)




