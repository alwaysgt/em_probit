The package is for linear approximation of a probit model. The data generating process for a probit model is simple

<p align="center">
  <img width="100" src="https://owgt.me/images/formula_1.png">
</p>

It provides a function to calculate the conditional expectation of $\Theta + \epsilon$ given $Y$ and $X \beta$. 
It's used in the EM algorithm for fitting a probit model.

Our package supports the cases in which the covariance of epsilon is not identity.The function for the conditional covariance of $\Theta + \epsilon$ is not provided at the moment.

## Usage 
We provides an example to fit a probit model with our packages.



