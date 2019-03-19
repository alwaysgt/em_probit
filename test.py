import em_probit
import numpy as np


'''
Generate a model and data
'''
# number of observation
n = 200
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
# The precision matrix
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
def linear_regression(X,y):
    return np.linalg.inv(X.T @ X)@X.T@y

em_probit_estimator = em_probit.EM_probit_normal(X,Y,np.eye(n))
beta = em_probit_estimator.EM(linear_regression)
l2_error = np.sum((beta - beta_0)**2)/np.sum(beta_0**2)
print("The relative squared error of our parameter estimator is {}".format(l2_error))

print("The difference might come from the simulation variance")

    
