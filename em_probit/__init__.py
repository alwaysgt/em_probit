#from .sampling import *
import numpy as np
import warnings
from ._sampling import sampling 

class EM_probit_normal:
    def __init__(self,X,y,Sigma = None):
        self.X = X 
        self.n,self.p = X.shape
        self.y = y
        if Sigma is None:
            self.Sigma = np.identity(self.n)
        else:
            self.Sigma = Sigma 
        
        # H is the inverse of the covariance matrix
        self.H = np.linalg.inv(self.Sigma)
        
        #Caching the calculated beta for warmup
        self.beta_cached = None


    def E_step(self,beta,sigma2):
        '''
        The goal is to regenerate the data y_hat
        where y_hat = X beta + epsilon
        '''
        y_hat_0 = self.X @ beta
        return sampling(self.y,y_hat_0,self.H)    
    
    def EM(self,estimator,n_iteration = 10):
        if self.beta_cached is None:
            beta = np.random.randn(self.p)
        else:
            beta = self.beta_cached
            
        sigma2 = self.sigma2_cached
        
        for i in range(n_iteration):
            y_E = self.E_step(beta,sigma2)
            beta = estimator(self.X,y_E)
            
        self.beta_cached = beta
        self.sigma2_cached = sigma2
        
        return beta

    