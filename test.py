import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

def rosen(x): 
    return sum(100*(x[1:] - (x[:-1])**2)**2 + (1 - x[:-1])**2)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
# res = minimize(rosen, x0=x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
# print(res.x)

# def rosen_der(x):
#     xm = x[1:-1]
#     xm_m1 = x[:-2]
#     xm_p1 = x[2:]
#     der = np.zeros_like(x)
#     der[1:-1] = 200*(xm - xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1 -xm)
#     der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1 - x[0])
#     der[-1] = 200*(x[-1] - x[-2]**2)
#     return der 

# def rosen_hess_p(x, p):
#     x = np.asarray(x)
#     Hp = np.zeros_like(x)
#     Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
#     Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] \
#                -400*x[1:-1]*p[2:]
#     Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
#     return Hp

# res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': True})
# print(res.x)

# res = minimize(rosen, x0, method='Newton-CG',
#                jac=rosen_der, hessp=rosen_hess_p,
#                options={'xtol': 1e-8, 'disp': False})
# print(res)

# bounds = Bounds([0, -0.5], [1.0, 2.0])
# linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])

from system_params import *

mode = np.asarray([1, 0, 0, 1, 1, 0, 1])
f_iU_0 = np.asarray([1, 1, 1, 1])
idx1=np.where(mode==1)[0]
Q = np.asarray([11, 11, 12, 7, 7, 4, 13])

def objective_func(f_iU):
    return np.sum(-Q[idx1]*f_iU*delta/F + V*psi*kappa*delta*(f_iU**3))

def obj_der(f_iU): 
    return -Q[idx1]*delta/F + V*psi*kappa*3*(f_iU**2)

def obj_hess(f_iU): 
    return V*kappa*psi*delta*6*f_iU

x = np.ones_like(idx1)
bounds = Bounds(x*0, x*Q[idx1]*F/delta)
linear_constraint = LinearConstraint(x, 0, f_u_max)

f_iU_0 = np.ones_like(idx1)
res = minimize(objective_func, f_iU_0, method='trust-constr', jac=obj_der, hess=obj_hess,
               constraints=[linear_constraint], options={'verbose': 1, 'disp': True}, bounds=bounds)

print(res.x)
value = res.fun
print(value)
print('Simulation finish!')



