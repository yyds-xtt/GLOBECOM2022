import numpy as np 
import pandas as pd 

from system_params import mu_gain, sigma_gain
from system_params import a_LOS, b_LOS
from system_params import xi, gamma 
from system_params import H_uav, g0 
from system_params import T, N 

from utils import * 

def channel_model():  
    '''
    generate channel gain in dB of 1 user 
    ''' 
    dist = np.random.randint(10, 100)

    theta = np.arctan(H_uav/dist)
    fading = np.random.normal(mu_gain, sigma_gain)
    # print(fading)

    p_LOS = 1./(1 + a_LOS*np.exp(-b_LOS*(theta - a_LOS)))

    # assume we omit the small scale fading 
    channel_gain = todB(p_LOS + xi*(1 - p_LOS)) + g0 + fading - todB((H_uav**2 + dist**2)**(gamma/2))

    # channel_gain = (p_LOS + xi*(1 - p_LOS))* dB(g0)* dB(fading)/((H_uav**2 + dist**2)**(gamma/2))
    # return todB(channel_gain)
    return channel_gain

# print(channel_model())

def gen_channel_gain(): 
    gain = np.zeros((T, N))
    import itertools
    for i, j in itertools.product(np.arange(T), np.arange(N)): 
        gain[i, j] = channel_model().copy()

    # df = pd.DataFrame
    # ({
    #     'dataA': gain
    # })
    # df.to_csv('channel_gain.csv')
    np.save('channel_gain', gain)
    return gain

gen_channel_gain()

channel_gain = np.load('channel_gain.npy')
print("load completed")