import numpy as np 

from system_params import *
from utils import * 

def channel_model():   
    dist = np.random.randint(10, 70, (N))

    theta = np.arctan(H_uav/dist)
    fading = np.random.normal(mu_gain, sigma_gain, (N))
    # print(fading)

    p_LOS = 1./(1 + a_LOS*np.exp(-b_LOS*(theta - a_LOS)))

    # assume we omit the small scale fading 
    channel_gain = todB(p_LOS + xi*(1 - p_LOS)) + g0 + fading - todB((H_uav**2 + dist**2)**(gamma/2))

    # channel_gain = (p_LOS + xi*(1 - p_LOS))* dB(g0)* dB(fading)/((H_uav**2 + dist**2)**(gamma/2))
    # return todB(channel_gain)
    return channel_gain

# print(channel_model())
