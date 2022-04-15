
import numpy as np
import matplotlib.pyplot as plt 

from system_params import *

class Position: 
    def __init__(self, r, varphi) -> None:
        self.r = r
        self.varphi = varphi 
        pass
        

class User: 
    def __init__(self, init_param) -> None:
        self.loc = []
        self.varphi = []

        self.loc.append(init_param[0])
        self.varphi.append(init_param[1])
        self.phi = init_param[2]
        self.velocity = init_param[3]

        pass

    def generate_channel_gain(self): 
        # generate the location 
        del_r = np.random.normal(self.velocity*delta, 1/3*self.velocity*delta, (T)) 
        del_varphi_arr = []
        for i_t in range(1, T): 
            # update delta_varphi and varphi angle 
            del_varphi = np.random.normal(self.phi - self.varphi[-1], np.pi/3)
            del_varphi_arr.append(del_varphi)
            self.varphi.append(self.phi + del_varphi)
            # update location
            self.loc.append(((self.loc[-1])**2 + (del_r[-1])**2 + 2*self.loc[-1]*del_r[-1]*np.cos(self.varphi[-1]))**(1/2))

        # calculate theta angle 
        theta = np.asarray([np.arctan(H_uav/loc) for loc in self.loc])

        # calculate the line of sight probability 
        p_LOS = 1./(1 + a_LOS*np.exp(-b_LOS*(theta - a_LOS)))

        # generate the small scale fading 
        h_tidle = np.random.normal(mu_gain, sigma_gain, size=(T)) # dB

        # calculate the channel gain 
        # channel_gain = (p_LOS + xi*(1 - p_LOS))*h_tidle*g0/((H_uav**2 + self.loc**2)**(gamma/2))
        gain_wo_fading_dB = todB(p_LOS + xi*(1 - p_LOS)) + g0 - todB((H_uav**2 + (np.array(self.loc)**2))**(gamma/2))
        gain_w_fading_dB = gain_wo_fading_dB + h_tidle
        x = [i for i in range(T)]

        plt.figure(0)
        plt.plot(x, gain_w_fading_dB, "-", x, gain_wo_fading_dB, '--')

        
        self.plot_location()
        plt.show()
        print('finish showing!')

        return gain_w_fading_dB
    
    def plot_location(self): 
        plt.figure(1)
        x = self.loc*np.cos(self.varphi)
        y = self.loc*np.sin(self.varphi)
        plt.plot(x, y, '-o')

if __name__ == '__main__':
    u = [110, np.pi/4, np.pi, 1.5] 
    user = User(u)
    user.generate_channel_gain()
    print('simulation finish!')



        
