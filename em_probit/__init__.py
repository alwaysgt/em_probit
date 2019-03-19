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


    def E_step(self,beta):
        '''
        The goal is to regenerate the data y_hat
        where y_hat = X beta + epsilon
        '''
        y_hat_0 = self.X @ beta
        return sampling(self.y,y_hat_0,self.H)    
    
    def EM(self,estimator,eps = 1e-3,max_iter = 300):
        if self.beta_cached is None:
            beta = np.zeros(self.p)
        else:
            beta = self.beta_cached
            
        for i in range(max_iter):
            y_E = self.E_step(beta)
            beta_new = estimator(self.X,y_E)
            if np.sum((beta- beta_new)**2)/np.sum(beta_new**2) < eps:
                beta = beta_new
                break
            beta = beta_new
        else:
            warn_info = "Maximum iterations {} reached and the optimization hasn't converged yet.".format(max_iter)
            warnings.warn(warn_info)
            
        self.beta_cached = beta        
        return beta

    