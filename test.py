import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

def rosen(x): 
    return sum(100*(x[1:] - (x[:-1])**2)**2 + (1 - x[:-1])**2)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
# res = minimize(rosen, x0=x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
# print(res.x)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm - xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1 -xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1 - x[0])
    der[-1] = 200*(x[-1] - x[-2]**2)
    return der 

def rosen_hess_p(x, p):
    x = np.asarray(x)
    Hp = np.zeros_like(x)
    Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
    Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] \
               -400*x[1:-1]*p[2:]
    Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
    return Hp

res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': True})
print(res.x)

res = minimize(rosen, x0, method='Newton-CG',
               jac=rosen_der, hessp=rosen_hess_p,
               options={'xtol': 1e-8, 'disp': False})
print(res)

# bounds = Bounds([0, -0.5], [1.0, 2.0])
# linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])

