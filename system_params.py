from utils import *

# System parameters
N = 10 # number of users
T = 100 # number of TS

W = mega(1) # Bandwidth mhz
# R = kilo(1) # packet size kb
R = 1 
V = 1e4 # Lyapunov 

# channel model 
H_uav = 100 # high of uav 
a_LOS = 9.16
b_LOS = 0.16
g0 = dB(-50)
xi = 0.2 # attenuation effect 
gamma = 2.7601 # path loss exponent 

kappa = 1e-27 
f_i_max = giga(0.5)
f_u_max = giga(10)
delta = mini(100)
psi = 10
p_i_max = dBm(20)
d_th = 15
F = 1000*R # CPU cycles / packet
