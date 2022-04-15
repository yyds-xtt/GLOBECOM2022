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
from system_params import *


def Algo1_NUM(mode, h, Q, L, V=20):

    ch_fact = 10**10   # scaling factors to avoid numerical precision problems
    d_fact = 10**6

    N0 = W*(10**(-17.4))*(10**(-3))  # noise power in watt

    N = len(Q)

    energy = np.zeros((N))

    f0_val = 0
    a_i = np.zeros((N))

    # uav computation frequency
    f_i = np.zeros((N))

    idx0 = np.where(mode == 0)[0]
    M0 = len(idx0)  # M0: number of local computation users

    if M0 == 0:
        f0_val = 0  # objective value of local computing user
    else:
        q0 = np.zeros((M0))
        f0 = np.zeros((M0))  # optimal local computing frequency
        for i in range(M0):
            tmp_id = idx0[i]
            q0[i] = Q[tmp_id]

            f_hat = np.minimum(f_i_max, q0[i]*F/delta)
            f0[i] = np.minimum(np.sqrt(q0[i]/3/F/V/kappa), f_hat)
            energy[tmp_id] = kappa*(f0[i]**3)
            f0_val = f0_val + energy[tmp_id] - q0[i]*f0[i]*delta/F
        # update resource allocation variable
        f_i[idx0] = f0
        # update local computation volume
        a_i = np.round(f_i*delta/F)

    idx1 = np.where(mode == 1)[0]
    M1 = len(idx1)
    f1_val = 0
    f2_val = 0
    f_iU = np.zeros((N))
    b_i = np.zeros((N))
    c_i = np.zeros((N))

    if M1 == 0:
        f1_val = 0  # objective value of remote offloading
    else:
        q1 = np.zeros((M1))
        l1 = np.zeros((M1))
        b1 = np.zeros((M1))  # optimal offloading volumn
        f1 = np.zeros((M1))  # optimal uav computation frequency
        for i in range(M1):
            tmp_id = idx1[i]
            q1[i] = Q[tmp_id]
            l1[i] = L[tmp_id]

            # objective value of remote offloading
            b_hat = np.minimum(q1[i], np.round(
                W*delta/R * np.log2(1 + p_i_max*h[i]/N0/W)))
            
            b1[i] = 0 if (q1[i] <= l1[i]) else np.maximum(0, np.minimum(
                np.round(W*delta/R * np.log2(h[i]*(q1[i] - l1[i])/(V*N0*R*np.log(2)))), b_hat))
            f1_val = b1[i]*(q1[i] - l1[i]) + V*(N0*W*delta/h[i])*(2**(b1[i]*R/W/delta) - 1)
        
        # update offloading volume 
        b_i[idx1] = b1
        
        # uav computation frequency

        def objective_func(f_iU):
            return np.sum(-L[idx1]*f_iU*delta/F + V*psi*kappa*delta*(f_iU**3))

        def obj_der(f_iU): 
            return -L[idx1]*delta/F + V*psi*kappa*3*(f_iU**2)

        def obj_hess(f_iU): 
            return V*kappa*psi*delta*6*f_iU

        x = np.ones_like(idx1)
        bounds = Bounds(x*0, x*L[idx1]*F/delta)
        linear_constraint = LinearConstraint(x, 0, f_u_max)

        f_iU_0 = np.ones_like(idx1)
        res = minimize(objective_func, f_iU_0, method='trust-constr', jac=obj_der, hess=obj_hess,
                    constraints=[linear_constraint], options={'verbose': 1, 'disp': True}, bounds=bounds)

        # update uav frequency 
        f_iU[idx1] = res.x
        c_i = np.round(f_iU*delta/F)

        f2_val = res.fun

    f_val = f1_val + f0_val + f2_val

    f_val = np.around(f_val, decimals=6)
    return f_val, a_i, b_i, c_i

# if __name__ == "__main__":
#     mode = np.asarray([1, 0, 0, 1, 1, 0, 1])
#     mu, sigma = 0, 0.1  # mean and standard deviation
#     h = np.abs(np.random.normal(mu, sigma, 7))
#     Q = np.asarray([11, 11, 12, 7, 7, 4, 13])
#     L = np.asarray([12, 14, 11, 2, 1, 6, 5])

#     t1, t2 = Algo1_NUM(mode, h, Q, L)
#     print('simulation finish!')
