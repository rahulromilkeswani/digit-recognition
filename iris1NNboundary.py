import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.markers import MarkerStyle
from scipy.spatial.distance import cdist
from scipy import stats

#  KNN function
def knn_predict(X_test, X_train, y_train, k=1):
    n_X_test = X_test.shape[0]  #return number of rows
    decision = np.zeros((n_X_test, 1))  #a zero column vector of size N_X_test
    for i in range(n_X_test):
        point = X_test[[i],:]  #selecting points as x,y co-ordinate
        #  compute euclidan distance from the point to all training data
        dist = cdist(X_train, point) #a vector of size equal to the training sample size
        #  sort the distance, get the index
        idx_sorted = np.argsort(dist, axis=0)
        #  find the most frequent class among the k nearest neighbour
        pred = stats.mode( y_train[idx_sorted[0:k]] )
        decision[i] = pred[0] 
    return decision

np.random.seed(1)

# Setup data
D = np.genfromtxt('iris.csv', delimiter=',')   #loads values from file
X_train = D[:, 0:2]   # feature extraction
y_train = D[:, -1]    # label extraction

# Setup meshgrid
x1, x2 = np.meshgrid(np.arange(2,5,0.01), np.arange(0,3,0.01)) #computes co-ordinates to set up a meshgrid
X12 = np.c_[x1.ravel(), x2.ravel()]  #array of size 90000 with values of [x1,x2]. Ravel returns flattened array

# Compute 1NN decision
k = 1
decision = knn_predict(X12, X_train, y_train, k) #function call, contains predicted class value for each point in grid

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#  plot decisions in the grid
decision = decision.reshape(x1.shape) #shaping back to form of a grid
plt.figure()
plt.pcolormesh(x1, x2, decision, cmap=cmap_light) #plotting decision boundaries

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, s=25) #plotting the training samples
plt.xlim(x1.min(), x1.max()) #setting limits of x axis
plt.ylim(x2.min(), x2.max()) #setting limits of y axis
plt.show()

sample_size_list = [10,20,30,50] #sample sizes
available_classes = [1,2,3] #possible classes
#sample_size_list = [10] #sample sizes

training_data_list = [] 
decision_list = []
modified_sample_sizes = []
for i in sample_size_list:
    temp_D = D.copy()
    random_selected_indices = np.array(np.random.choice(np.arange(len(D)),size=i)) #selecting random indices
    for j in random_selected_indices:
        existing_class = temp_D[j,-1] #actual class for the current point
        selectable_classes = available_classes[:]
        selectable_classes.remove(existing_class) 
        temp_D[j,-1]=np.random.choice(selectable_classes)  #swapping the actual class with either of the remaining classes
    modified_sample_sizes.append(temp_D)  
    train_points = temp_D[:,0:2] #extracting co-ordinates
    train_labels = temp_D[:,-1] #extracting labels
    decision = knn_predict(X12, train_points, train_labels,3)
    decision = decision.reshape(x1.shape) #shaping back to form of a grid
    decision_list.append(decision) 
    training_data_list.append(temp_D)

#plotting the decisions   
plt.figure()
for i in range(len(decision_list)):
    plt.subplot(2,2,i+1)
    plt.pcolormesh(x1,x2,decision_list[i],cmap=cmap_light)
    plt.scatter(training_data_list[i][:,0], training_data_list[i][:,1], c=training_data_list[i][:,-1], cmap=cmap_bold, s=25) #plotting the training samples
    plt.xlim(x1.min(), x1.max()) #setting limits of x axis
    plt.ylim(x2.min(), x2.max()) #setting limits of y axis
    plt.title("m = "+ str(sample_size_list[i]))

plt.show() 

for sample in modified_sample_sizes:
    m_predicted_list=[]
    for i in range(0,len(sample)) : 
        temp_sample = sample [:]
        test_point = np.array(temp_sample[i][0:2]).reshape(-1,2) #extracting point and reshaping it
        train_points = np.delete(temp_sample,i,axis = 0) #deleting current point from the whole list
        train_points_coordinates = train_points[:,0:2] #remaining points after leaving out the test point
        train_points_labels = train_points[:,-1] #labels of the remaining train points
        dist = cdist(train_points_coordinates, test_point) #calculating the distance of test point against train points
        idx_sorted = np.argsort(dist, axis=0) #sorting the distances
        predicted_label = stats.mode(train_points_labels[idx_sorted[0:3]])   #picking the mode label of the nearest three points   
        m_predicted_list.append(predicted_label[0])
    reshaped_predicted_list=np.concatenate(m_predicted_list,axis=1)
    error_rate = (reshaped_predicted_list!=sample[:,-1]).mean() #calculating error
    print(error_rate*100)    


        