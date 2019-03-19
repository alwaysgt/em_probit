The data generating process for a probit model could be written as 

<p align="center">
  <img width="100" src="https://owgt.me/images/formula_1.png">
</p>

The package provides a function to calculate the conditional expectation of $\Theta + \epsilon$ given $Y$ and $X \beta$, which is used in the EM algorithm for fitting a probit model. Our package supports the scenarios in which the covariance of epsilon is not the identity.

It also provides a function to fit a probit model, in which \epsilon has i.i.d. coordinates.



## Usage 
We provides an example using EM algorithm to fit a probit model with our packages, given data X,y. We use the package for fitting a probit model. We also provide a high-level api for the fitting. See `test.py`


```python
import em_probit
import numpy as np

'''
Fitting the model
'''
#Define linear regression:
def linear_regression(X,y):
    return np.linalg.inv(X.T @ X)@X.T@y

#initialize estimator 
beta = np.zeros(p)
#fitting the model
while 1:
    #Calculate the posterior mean, given the current estimator, H is the precision matrix of epsilon
    Y_hat = em_probit.sampling(Y,X@beta,H)
    #Get a new estimator
    beta_new = linear_regression(X,Y_hat)
    #Check the convergence
    if np.sum((beta- beta_new)**2)/np.sum(beta_new**2) < 1e-3:
        beta = beta_new
        break
    beta = beta_new
# output the squared error of the estimator
l2_error = np.sum((beta - beta_0)**2)/np.sum(beta_0**2)
print("The relative squared error of our parameter estimator is {}".format(l2_error))
```

## Purpose of the package
The package is written for the thesis Spectral Deconfounding on Generalized Linear Models ( Here's the [introduction page](https://owgt.me/deconfounding_lava.html)). We try to exploit the linear structure of GLMs to implement the deconfounding method, which is originally disigned for linear models.
