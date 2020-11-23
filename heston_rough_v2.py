#=============================================================================================================================
#   Rough Heston Model - Heston with Hurst Coefficient (H<0.5). In this code, alpha = 1 + H - 0.5. For alpha = 0.6, H = 0.1.
#   Parameters taken from paper: Broadie et al, Exact simulation of stochastic volatility and other affine jump
#                                diffusion processes, Journal of Operations Research, 2006
#=============================================================================================================================

import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial
from scipy.integrate import trapz,simps

class base_heston():
    """
    Base class for the Heston process:
    r = interest rate
    rho = correlation between stock and variance noise
    theta = long run average volatility
    sigma = volatility of volatility
    kappa = mean reversion coefficient for the variance process
    mu = drift
    K = strike price
    alpha = 1 + H - 0.5 , Hurst Coefficient
    Lambda = Parameter from the Rough Heston Model
    """
    
    def __init__(self, mu=0.1, rho=0, sigma=0.2, theta=-0.1, kappa=0.1, K = 100, alpha=0.6,lamb = 2):
        self.mu = mu
        self.K = K
        if (np.abs(rho)>1):
            raise ValueError("|rho| must be <=1")
        self.rho = rho
        if (theta<0 or sigma<0 or kappa<0):
            raise ValueError("sigma,theta,kappa must be positive")
        else:
            self.theta = theta
            self.sigma = sigma
            self.kappa = kappa
            self.alpha = alpha
            self.lamb = lamb
    
    def integrate(self, S0, v0, N, T=1):
        
        """
        Numerical integration using the Euler method
        N = number of time steps
        T = Unit time
        Returns two arrays S (price) and v (volatility). 
        """

        #=========================================================================================
        #   Used the following source to create correlated random samples
        #   https://scipy-cookbook.readthedocs.io/items/CorrelatedRandomSamples.html
        #=========================================================================================
        
        MU = np.array([0, 0])
        COV = np.matrix([[1, self.rho], [self.rho, 1]])
        W = ss.multivariate_normal.rvs( mean=MU, cov=COV, size=N-1 )
        W_S = W[:,0]                   # Stock Brownian motion:     W_1
        W_v = W[:,1]                   # Variance Brownian motion:  W_2

        # Initialize vectors
        T_vec, dt = np.linspace(0,T,N, retstep=True )
        dt_sq = np.sqrt(dt)
        
        X0 = S0                 #Boundary condition for X
        v = np.zeros(N)                 #Initialize volatility
        v[0] = v0                       #Boundary condition for volatility
        X = np.zeros(N)                 #Initialize stock array
        X[0] = X0                       #Boundary condition for stock value

        # Generate paths using Euler Integration method
        for t in range(0,N-1):
            v_sq = np.sqrt(v[t])
            if(t>=1):
                x=np.arange(0,(t)*dt,(t)*dt/100)    # Create a dummy variable for integration of time, take time upto t
                vinterp = np.interp(x,range(0,t)*dt,v[:t]) # interpolate the volatility upto time t on x
                vsqinterp = np.sqrt(vinterp)    # Square root of the interpolated volatility
                v[t+1] = np.maximum(v0 + 1/gamma(self.alpha)*simps(((t+1)*dt-x)**(self.alpha-1)*self.lamb*(self.theta-vinterp),x) + 1/gamma(self.alpha)*simps(((t+1)*dt-x)**(self.alpha-1)*self.lamb*self.sigma * vsqinterp * dt_sq * W_v[t],x),0)
            else:
                v[t+1] = v0
            #Update the underlying based on volatility at previous time step
            X[t+1] = X[t] + (self.mu)*X[t]*dt + v_sq * dt_sq *X[t]* W_S[t]         #from paper, Euler
            #X[t+1] = X[t] * np.exp((self.mu - 0.5*v[t])*dt + v_sq * dt_sq * W_S[t] ) #same results as paper
        return X, v

    #Payoff for call option - at the end of the time period
    def payoff_call(self,S):
        return np.maximum(S-self.K,0)

    #Payoff for put option - at the end of the time period
    def payoff_put(self,S):
        return np.maximum(self.K-S,0)

#===============================================================================================
#           Actual implementation
#===============================================================================================
#Parameters are taken from paper
N = 1000                                                      # time points 
T = 1                                                        # units in time, here it is in years as the riskless rate is defined as per year
T_vec, dt = np.linspace(0,T,N, retstep=True )                 # time vector and time step
S0 = 100                                                      # initial stock price
K = 100                                                       # Strike price
v0 = 0.010201                                                 # initial volatility
r = 0.0319                                                    # Riskless Rate of return 
mu = r; rho = -0.7; kappa = 6.21; theta = 0.019; sigma = 0.61    #values for different 
assert(2*kappa * theta <= sigma**2)                            # Condition from the paper is asserted, a method to check if the parameters are correct
num_sims = 1000                                            #Monte Carlo simulation paths
call_payoff = 0.0                                             #Payoff for call option
put_payoff = 0.0                                              #Payoff for put option

np.random.seed(seed=1111)
#Initialize the class
Hest = base_heston(mu=mu, rho=rho, sigma=sigma, theta=theta, kappa=kappa, K=K)

#Monte Carlo simulation - create the different paths
for i in range(0,num_sims):
    S, V = Hest.integrate(S0, v0, N, T)
    call_payoff += Hest.payoff_call(S[-1])
    put_payoff += Hest.payoff_put(S[-1])
    print("Completed simulation path number: ",i)

#calculate payoff from monte-carlo simulations
call_payoff = call_payoff/num_sims*np.exp(-r*T)
put_payoff = put_payoff/num_sims*np.exp(-r*T)

#Print the results
print("Price of the call option = %.2f" %(call_payoff))
print("Price of the put option = %.2f" %(put_payoff))


