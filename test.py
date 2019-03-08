import em_probit
import numpy as np



y = np.array([0,0,1,0,0])
mu = np.array([-1.,5,.6,-0.9,1.5])
H = np.eye(5)

a = em_probit.sampling(y,mu,H)
print(a)