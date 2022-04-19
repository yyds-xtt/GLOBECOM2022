from utils import *

# System parameters
N = 10 # number of users

time = 120 # total simulation time
delta = mini(100)
T = int(time/delta) # number of TSs

W = 0.2*mega(1) # Bandwidth mhz
R = kilo(1) # packet size kb
V = 1e4 # Lyapunov 

# channel model 
H_uav = 50 # high of uav 
a_LOS = 9.16
b_LOS = 0.16
g0 = -48 # channel gain reference dB 
xi = 0.2 # attenuation effect 
gamma = 2.7601 # path loss exponent 

mu_gain = 0 # dB fading channel power gain 
var_gain = 4 # fading channel variance 
sigma_gain = 2 # sqrt(var_gain)

kappa = 1e-27 
f_i_max = giga(0.5)
f_u_max = giga(20)

psi = 0.1
p_i_max = dBm(20)
d_th = 15
F = 500*R # CPU cycles / packet

CHFACT = 10**12       # The factor for scaling channel value
scale_delay = 1e3