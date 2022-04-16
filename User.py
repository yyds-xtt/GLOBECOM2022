
from re import S
from turtle import update
import numpy as np
import matplotlib.pyplot as plt 

from system_params import *

class Position: 
    def __init__(self, r, varphi):
        self.r = r
        self.varphi = varphi
        pass

    def update(self, dr, dphi):
        r = self.r  
        r2 = r**2
        dr2 = dr**2
        nr = np.sqrt(r2 + dr2 + 2*r*dr*np.cos(dphi))
        angle = np.arccos((r2 + nr**2 - dr2)/2/r/nr) + self.varphi 
        return Position(nr, angle)

    # @property
    def varphi(self): 
        return self.varphi
    # @property
    def r(self): 
        return self.r
    
    def get_decart_pos(self): 
        return (self.r*np.cos(self.varphi), self.r*np.sin(self.varphi))


class User: 
    def __init__(self, init_param) -> None:
        self.loc = []
        self.loc.append(Position(init_param[0], init_param[1]))
        self.phi = init_param[2]
        self.velocity = init_param[3]

        pass

    def generate_channel_gain(self): 
        # generate the location 
        dr = np.random.normal(self.velocity*delta, 1/3*self.velocity*delta, (T)) 
        dvarphi_arr = []
        for i_t in range(1, T): 
            current_position = self.loc[-1]
            # update delta_varphi and varphi angle 
            dvarphi = np.random.normal(self.phi - current_position.varphi, np.pi/3)
            dvarphi_arr.append(dvarphi)
            next_position = current_position.update(dr[i_t], dvarphi)
            self.loc.append(next_position)

        # calculate theta angle 
        theta = np.asarray([np.arctan(H_uav/position.r) for position in self.loc])

        # calculate the line of sight probability 
        p_LOS = 1./(1 + a_LOS*np.exp(-b_LOS*(theta - a_LOS)))

        # generate the small scale fading 
        h_tidle = np.random.normal(mu_gain, sigma_gain, size=(T)) # dB

        # calculate the channel gain 
        # channel_gain = (p_LOS + xi*(1 - p_LOS))*h_tidle*g0/((H_uav**2 + self.loc**2)**(gamma/2))
        distance = [(H_uav**2 + (position.r**2))**(gamma/2) for position in self.loc]
        gain_wo_fading_dB = todB(p_LOS + xi*(1 - p_LOS)) + g0 - todB(distance)
        gain_w_fading_dB = gain_wo_fading_dB + h_tidle
        x = [i for i in range(T)]

        plt.figure(0)
        plt.plot(x, gain_w_fading_dB, "-", x, gain_wo_fading_dB, '--')
        # self.plot_location()
        plt.show()
        print('finish showing!')

        return gain_w_fading_dB
    
    def plot_location(self): 
        
        xt = np.zeros((T))
        yt = np.zeros((T))
        for it, loc in enumerate(self.loc): 
            xt[it], yt[it] = loc.get_decart_pos()   
        print('start plotting') 
        plt.figure(1)
        plt.plot(xt, yt, '-')

if __name__ == '__main__':
    u = [110, np.pi/4, np.pi, 1.5] 
    user = User(u)
    user.generate_channel_gain()
    print('simulation finish!')



        
