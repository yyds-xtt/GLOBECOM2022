    # -*- coding: utf-8 -*-
"""
Agorithm 1 of solving th optimal resource allocation problem (P4) in Sev.IV.B given offloading decisions in (P1)

Input: binary offloading mode, channel, weighting parameter, data queu length, current data arrival, virtual energy queue length  

Output: the optimal objective, the computation rate and energy consumption of all users

Created on Sat May 9 2020
@author: BI Suzhi
"""  
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from utils import *
import system_params


def Algo1_NUM(mode,h,Q,L, V=20):
   
    ch_fact = 10**10   # scaling factors to avoid numerical precision problems
    d_fact = 10**6
    F =  system_params.F # number of cpu cycles for processing 1 bit data
    W = system_params.W    # bandwidth MHz
    kappa = system_params.kappa
    
    N0 = W*d_fact*(10**(-17.4))*(10**(-3))*ch_fact # noise power in watt
    p_i_max = system_params.p_i_max # maximum transmit power 100mW
    f_i_max = system_params.f_i_max # maximum local computing frequency 100MHz
    delta = system_params.delta
    psi = system_params.psi
    R = system_params.R
    f_u_max = system_params.f_u_max

    N = len(Q)

    energy = np.zeros((N));
    rate = np.zeros((N));
    f0_val = 0;

    # uav computation frequency 
    
            
    idx0 = np.where(mode==0)[0]
    M0 = len(idx0) #M0: number of local computation users
    
    if M0==0:
        f0_val =0 # objective value of local computing user
    else:
        q0 = np.zeros((M0))
        f0 = np.zeros((M0)) # optimal local computing frequency
        for i in range(M0): 
            tmp_id = idx0[i]
            q0[i] = Q[tmp_id]

            f_hat = np.minimum(f_i_max, q0[i]*F/delta)
            f0[i] = np.minimum(np.sqrt(q0[i]/3/F/V/kappa),f_hat)
            energy[tmp_id] = kappa*(f0[i]**3)
            f0_val =  f0_val + energy[tmp_id] - q0[i]*f0[i]*delta/F;                         
            
    idx1=np.where(mode==1)[0]
    M1 = len(idx1) 
    f1_val = 0
    f2_val = 0
    
    
    if M1==0:
        f1_val = 0 # objective value of remote offloading
    else:
        q1 = np.zeros((M1))
        l1 = np.zeros((M1))
        b1 = np.zeros((M1)) # optimal offloading volumn
        f1 = np.zeros((M1)) # optimal uav computation frequency 
        for i in range(M1):
            tmp_id = idx1[i]
            q1[i] = Q[tmp_id]
            l1[i] = L[tmp_id]

            # objective value of remote offloading
            b_hat = np.minimum(q1[i], np.round(W*delta/R * np.log2(1 + p_i_max*h[i]/N0)))
            b1[i] = np.minimum(np.round(W*delta/R * np.log2(h[i]*(q1[i] - l1[i])/(V*N0*R*np.log(2)))), b_hat)
            f1_val = b1[i]*(l1[i] - q1[i]) + V*(N0*W*delta/h[i])*(2**(b1[i]*R/W/delta) - 1)

        # uav computation frequency
        # nested objective function
        def objective_func(f_iU):
            return np.sum([-Q[idx1]*f_iU*delta/F] + V*psi*kappa*delta*(f_iU**3))
        
        x = np.ones_like(idx1)
        bounds = Bounds(x*0, x*Q[idx1]*F/delta)
        linear_constraint = LinearConstraint([x.T], 0, f_u_max)

        f_iU_0 = np.ones_like(idx1)
        res = minimize(objective_func, f_iU_0, method='trust-constr', 
            constraints=[linear_constraint], options={'verbose': 1, 'disp': True}, bounds=bounds)

        print(res.f_iU)
        f2_val = res.fun
    
    f_val = f1_val + f0_val + f2_val

    
    f_val = np.around(f_val, decimals=6)     
    rate =  np.around(rate, decimals=6)
    energy =  np.around(energy, decimals=6) 
    

    return f_val,rate,energy

mode = np.asarray([1, 0, 0, 1, 1, 0, 1]).T
mu, sigma = 0, 0.1 # mean and standard deviation
h = np.random.normal(mu, sigma, 7).T
Q = np.random.randint(14, size=(1, 7)).T
L = np.random.randint(20, size=(1, 7)).T

t1, t2, t3 = Algo1_NUM(mode, h, Q, L)    
      

