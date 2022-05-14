from utils import *

# System parameters
N = 10 # number of users

time = 10 # total simulation time
delta = mini(100) 
 
T = int(time/delta) # number of TSs
n = T         # number of time frames
<<<<<<< HEAD
W = mega(1) # Bandwidth mhz
R = kilo(1) # packet size kb
V = 2*1e4 # Lyapunov
=======
W = 0.1*N*mega(1) # Bandwidth mhz

R = kilo(1) # packet size kb
V = 5*1e5 # Lyapunov
>>>>>>> 642a114 (code 1405)

# channel model 
H_uav = 50 # high of uav 
a_LOS = 9.16
b_LOS = 0.16
g0 = -50 # channel gain reference dB 
xi = 0.2 # attenuation effect 
gamma = 2.7601 # path loss exponent 
N0 = dBm(-174)

mu_gain = 0 # dB fading channel power gain 
var_gain = 4 # fading channel variance 
sigma_gain = 2 # sqrt(var_gain)

kappa = 1e-26
f_i_max = giga(0.5)
f_u_max = giga(1.5)

psi = 0.04
p_i_max = dBm(20)
F = 500*R # CPU cycles / packet

scale_delay = 20
d_th = 5

lambda_param = np.round(0.1*1e6/R)
# the quantization mode could be 'OP' (Order-preserving) or 'KNN' or 'OPN' (Order-Preserving with noise)
decoder_mode = 'OPN'
Memory = 1024          # capacity of memory structure
Delta = 32             # Update interval for adaptive K

CHFACT = 1e12     # The factor for scaling channel value
QFACT = 1/150     # The factor for scaling channel value
LFACT = 1/200    # The factor for scaling channel value
DFACT = 1/3     # The factor for scaling channel value

mode = 'test'
# mode = 'ntest'

######## save and load params #########

dataA= np.load('dataA.npy')
channel_gain = np.load('channel_gain.npy')
######################################

window_size = 10
opt_mode_arr = ['LYDROO', 'bf']
opt_mode = opt_mode_arr[0]

no_nn_inputs = 4

comparison_flag = False 
