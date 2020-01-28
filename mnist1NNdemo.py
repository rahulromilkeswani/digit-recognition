#  Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance
import random
import numpy as np
import matplotlib.pyplot as plt
import mnist

def sqDistance(p, q, pSOS, qSOS):
    #  Efficiently compute squared euclidean distances between sets of vectors
    #  Compute the squared Euclidean distances between every d-dimensional point
    #  in p to every d-dimensional point in q. Both p and q are
    #  npoints-by-ndimensions. 
    #  d(i, j) = sum((p(i, :) - q(j, :)).^2)

    d = np.add(pSOS, qSOS.T) - 2*np.dot(p, q.T)
    return d
def trainSizeModerator(train_set_size):
  np.random.seed(1)
  Xtrain, ytrain, Xtest, ytest = mnist.load_data() #loading the dataset
  train_size = train_set_size
  test_size  = 10000
  Xtrain = Xtrain[0:train_size]
  ytrain = ytrain[0:train_size]
  Xtest = Xtest[0:test_size]
  ytest = ytest[0:test_size]

  #  Precompute sum of squares term for speed
  XtrainSOS = np.sum(Xtrain**2, axis=1, keepdims=True)   #computes sum of squares of element in each row
  XtestSOS  = np.sum(Xtest**2, axis=1, keepdims=True)    #computes sum of squares of element in each row

  #  fully solution takes too much memory so we will classify in batches
  #  nbatches must be an even divisor of test_size, increase if you run out of memory 
  if test_size > 1000:
    nbatches = 50
  else:
    nbatches = 5

  batches = np.array_split(np.arange(test_size), nbatches)
  ypred = np.zeros_like(ytest)

  #  Classify
  for i in range(nbatches):
      dst = sqDistance(Xtest[batches[i]], Xtrain, XtestSOS[batches[i]], XtrainSOS)  #computes eucledian distance between test and train data
      closest = np.argmin(dst, axis=1)   #picks index of the minimum distance
      ypred[batches[i]] = ytrain[closest] #assigns the value(label) of the minimum index to the predicted array

  #  Report
  errorRate = (ypred != ytest).mean()  #taking average of errors between actual and predcited labels
  #errorString = 'Error Rate: {:.2f}%\n'.format(100*errorRate)#formatting the output
  return errorRate*100

  #  image plot
  #plt.imshow(Xtrain[0].reshape(28, 28), cmap='gray')   #plotting the first training example 
  #print(ytrain[0])
  #plt.show()

#  Set training & testing 

# Q1:  Plot a figure where the x-axis is number of training
#      examples (e.g. 100, 1000, 2500, 5000, 7500, 10000), and the y-axis is test error.
np.random.seed(1)
Xtrain, ytrain, Xtest, ytest = mnist.load_data()
train_values = [100,1000,2500,5000,7500,10000]
error_rates_nn = []
for i in range(0,len(train_values)): 
  error_rates_nn.append(trainSizeModerator(train_values[i]))

# Q2:  plot the n-fold cross validation error for the first 1000 training training examples

Xtrain, ytrain, Xtest, ytest = mnist.load_data()
Xtrain = Xtrain[0:1000]
ytrain = ytrain[0:1000]
split_sizes=[3,10,50,100,1000] 
error_rates_cv=[]

for i in range(0,len(split_sizes)):
    predicted_labels=[]
    split_samples = np.array_split(np.arange(len(Xtrain)),split_sizes[i]) #splitting up into equal sub-lists
    for j in range(0,len(split_samples)):
      #choosing 1 validation set and remaining as training set
       validation_set = Xtrain[split_samples[j]] 
       validation_labels = ytrain[split_samples[j]]
       train_set = np.delete(Xtrain,split_samples[j],axis = 0)
       train_labels = np.delete(ytrain,split_samples[j],axis = 0)
       #calculating Sum of squares
       XtrainSOS = np.sum(train_set**2, axis=1, keepdims=True)
       validationSOS  = np.sum(validation_set**2, axis=1, keepdims=True)
       dst = sqDistance(validation_set, train_set, validationSOS, XtrainSOS) #calculating distance of validation set against training set
       closest = np.argmin(dst, axis=1) #sorting the distances and picking the closest
       predicted_labels.append(train_labels[closest]) #assigning label of the nearest point
    predicted_list=np.concatenate(predicted_labels,axis=0)
    errorRate = (predicted_list != ytrain).mean() #calculating error rate
    error_rates_cv.append(errorRate*100)

#plotting the graph
fig = plt.figure()
nn_plot = fig.add_subplot(1, 2, 1)
cv_plot = fig.add_subplot(1, 2, 2)
nn_plot.plot(train_values,error_rates_nn)
nn_plot.set_xlabel('Number of Training Samples')
nn_plot.set_ylabel('Test Error Percentage')
nn_plot.set_title('1-Nearest Neighbor')
cv_plot.plot(split_sizes,error_rates_cv)
cv_plot.set_xlabel('Number of Folds')
cv_plot.set_ylabel('Error Rate')
cv_plot.set_title('N Fold Cross Validation')
plt.show()