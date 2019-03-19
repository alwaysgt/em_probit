The data generating process for a probit model is 

<p align="center">
  <img width="100" src="https://owgt.me/images/formula_1.png">
</p>

The package provides linear approximation  and fitting for probit models. More exactly, the package provides a function to calculate the conditional expectation of $\Theta + \epsilon$ given $Y$ and $X \beta$, which used in the EM algorithm for fitting a probit model. It also provides a function to fit probit models.

Our package supports the cases in which the covariance of epsilon is not identity.The function for the conditional covariance of $\Theta + \epsilon$ is not provided at the moment.

## Usage 
We provides an example to fit a probit model with our packages.


```python
import em_probit
import numpy as np


'''
Generate a model and data
'''
# number of observation
n = 300
# number of variable
p = 5
# Design matrix 
X = np.random.randn(n,p)
# true parameter
beta_0 = np.random.randn(p)
# intermediate variable \Theta
Theta = X@beta_0
# dependent variable 
Y = ((Theta + np.random.randn(n) )> 0).astype(np.long)
# 
H = np.eye(n)



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
    Y_hat = em_probit.sampling(Y,X@beta,H)
    beta_new = linear_regression(X,Y_hat)
    if np.sum((beta- beta_new)**2)/np.sum(beta_new**2) < 1e-3:
        beta = beta_new
        break
    beta = beta_new
# output the squared error of the estimator
l2_error = np.sum((beta - beta_0)**2)/np.sum(beta_0**2)
print("The relative squared error of our parameter estimator is {}".format(l2_error))

'''
The fitting method is encapsulated in our module
'''

em_probit_estimator = em_probit.EM_probit_normal(X,Y,np.eye(n))
beta = em_probit_estimator.EM(linear_regression)
l2_error = np.sum((beta - beta_0)**2)/np.sum(beta_0**2)
print("The relative squared error of our parameter estimator is {}".format(l2_error))

print("The difference might come from the simulation variance")
```
