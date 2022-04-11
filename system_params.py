from utils import *

# System parameters
N = 5 # number of users
T = 100 # number of TS
F = 500 # CPU cycles / packet
W = mega(1) # Bandwidth mhz
R = kilo(0.1) # packet size kb
V = 20 # Lyapunov 
kappa = 10e-27 
f_i_max = giga(0.5)
f_u_max = giga(10)
delta = mini(100)
psi = 10
p_i_max = dBm(20)
