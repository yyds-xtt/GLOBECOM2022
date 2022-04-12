from utils import *

# System parameters
N = 5 # number of users
T = 100 # number of TS

W = mega(1) # Bandwidth mhz
R = kilo(300) # packet size kb
V = 20 # Lyapunov 
kappa = 1e-27 
f_i_max = giga(0.5)
f_u_max = giga(10)
delta = mini(100)
psi = 10
p_i_max = dBm(20)
d_th = 15
F = 500*R # CPU cycles / packet
