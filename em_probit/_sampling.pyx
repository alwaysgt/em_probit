from libc.stdlib cimport rand, RAND_MAX
from scipy.special.cython_special cimport ndtr,ndtri
import numpy as np
cimport numpy as np
np.import_array()
 

#Generate a uniform random variable on [0,1]
cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX

#Generate a normal random variable conditional on X \geq a 
cdef double truncated_norm_inf(double a): 
    cdef double temp  = ndtr(a)
    return ndtri(temp + (1-temp)*random_uniform())

#Generate a normal random variable conditional on X \leq b 
cdef double truncated_norm_minusinf(double b):
    cdef double temp  = ndtr(b)
    return ndtri(temp* random_uniform())


#Generate a normal random variable with a given mean and given standard deviation conditional on the sign y.
cdef double rvs(int y,  double mean, double sd):
    cdef double a
    if y == 1:
        a = (0. - mean)/sd
        return truncated_norm_inf(a)*sd + mean
    else:
        a = (0. - mean)/sd
        return truncated_norm_minusinf(a)*sd + mean


'''
y is a vector of signs
H is the inverse of the covariance matrix
mu is the estimated theta

We average the simulated mean and covariance in the end 
'''
cpdef sampling(np.ndarray[long,ndim = 1] y,np.ndarray[double, ndim =1] mu,np.ndarray[double, ndim =2 ] H,int n_simulating = 1000, int n_warm_start = 200):
        cdef int n =  H.shape[0]
 
        cdef np.ndarray[double] y_current  = np.zeros(n,np.double)
        cdef np.ndarray[double] y_sum = np.zeros(n,np.double)
        cdef double variance = 0
        cdef double mean = 0
        cdef int i,j

        for j in range(n_simulating + n_warm_start):
            for i in range(n):
                y_current[i] = mu[i]
                variance = 1/H[i,i]
                mean = mu[i] - variance * np.dot(H[i,:], y_current -  mu  )
                y_current[i] = rvs(y[i],mean,np.sqrt(variance))
                if np.abs(y_current[i]) == np.inf:
                    y_current[i] = 0
                
                
            if j >= n_warm_start:
                y_sum += y_current
                
        return y_sum/n_simulating
    
    