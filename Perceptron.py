import numpy as np


class Perceptron:
    """
    Perceptron Class

    Parameters
    __________

    eta : float 
        This is the learning rate (between 0.0 and 1.0)

    n_iter : int 
        This is the number of passes over the training set.

    random_state : int
        This is a random number generator seed for random weight initialization.

    Attributes
    __________
    2_ : 1d-array
        A 1 dimensional array with weights after fitting.
    
    b_ : Scalar
        This is the bias after fitting.

    errors_ : list
        This is the number of misclassifications in every epoch.

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1): 
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        __________

        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        __________

        self : object
        """

        """
        This is the initil weight vector. it contains small random numbers drawn from a normal dist with a std deviation of 0.01. 
        Using a small scale of 0.01 ensures the weights are small but not identical, and avoids setting the weights to 0, which would results in only changing the scale of the weight vector and not the direction.
        Effectively, this means the decision boundary will not change  
        rgen is a Numpy random  number generator that we seeded with a random seed so that we can reproduce previous results if required. 
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]) 

       
        self.b_ = np.float_(0.)
        self.errors_ = []

        """
        For each epoch up to n_inter, iterate through each training sample, compute the prediction error (difference between the actual and predicted values) and update the 
        weights and bias accordingly. the update is proportional to the error times the learning rate.
        """
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi)) # n(target - prediction)
                self.w_ += update * xi # multiply the adjusted magnitude by the feature value of x 
                self.b_ += update # add the adjusted magnitude to the bias
                errors += int(update != 0.0) # Keep track of the number of errors in each epoch 
            self.errors_.append(errors)
        return self
        

    """
    This function calcs the weighted sum of the input features (x) and the bias (b_) which is the net input to the perceptron 
    """
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    """
    this function returns the class label after applying the threshold of 0.0. If the net input is greater than 0.0, it returns 1, otherwise it returns 0.
    """
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        
    