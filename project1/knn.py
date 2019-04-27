"""
K Nearest Neighbour
"""

import numpy as np
from scipy import stats


class KNN(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        #########################################################################
        # TODO:     WRITE CODE FOR THE FOLLOWING                                #
        # Compute the L2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # a distance matrix.                                                    #
        #                                                                       #
        #  Implement this function using only basic array operations and        #
        #  NOT using functions from scipy: (scipy.spatial.distance.cdist).      #
        #                                                                       #
        #########################################################################
        
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        test_sum = np.sum(np.square(X), axis=1,keepdims=True).reshape(num_test,1)  # num_test x 1
        train_sum = np.sum(np.square(self.X_train),axis=1,keepdims=True).reshape(1,num_train)  # 1 x num_train
        inner_product = np.dot(X, self.X_train.T)  # num_test x num_train
        dists = np.sqrt(-2 * inner_product + test_sum.reshape(-1,1) + train_sum) 
        #(a-b)^2=a^2+b^2-2ab
        #########################################################################
        # TODO:     WRITE CODE FOR THE FOLLOWING                                #
        # Use the distance matrix to find the k nearest neighbors for each      #
        # testing points. Break ties by choosing the smaller label.             #
        #                           n                                              #
        # Try to implement it without using loops (or list comprehension).      #
        #                                                                       #
        #########################################################################
        
       # method with loop!!!
        y_pred=np.zeros(num_test)
        for i in range(num_test):
            closest_y=[]
           #np.argsort returns ascending value's location <<index>>
           #np.bincount:get the number of each value from closest_y value
           #np.argmax:returns the location of the maximum number
            y_indicies=np.argsort(dists[i,:])  
            closest_y=self.y_train[y_indicies[:k]]   
            y_pred[i]=np.argmax(np.bincount(closest_y))
            
        """
        method without loop!!! 
        y_pred=np.zeros(num_test)
        closest_y=[]
        #np.argsort returns ascending value's location <<index>>
        #np.bincount:get the number of each value from closest_y value
        #np.argmax:returns the location of the maximum number
        y_indicies=np.argsort(dists,axis=1) 
        closest_y=self.y_train[y_indicies[:,0:k]].reshape(num_test,k)
        #y_pred=np.argmax(np.bincount(np.squeeze(closest_y))) 
        y_pred=np.argmax(np.apply_along_axis(np.bincount,1,closest_y),axis=1) 
        """

        #########################################################################
        #                           END OF YOUR CODE                            #
        #########################################################################

        return y_pred
