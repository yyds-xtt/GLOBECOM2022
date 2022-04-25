import numpy as np
import matplotlib.pyplot as plt 

from system_params import *

rng = np.random.default_rng(seed=42)  

class User: 
    def __init__(self, 
    init_location, 
    arrival_rate):
        
        self.loc = []
        self.gain_dB = []
        self.channal_gain_wo_fading_dB = []
        
        self.arrival_rate = arrival_rate
        self.Q = np.zeros((no_slots))
        
        self.generate_channel_gain(init_location)

    def update(self, dr, dphi):
        r = self.r  
        r2 = r**2
        dr2 = dr**2
        next_r = np.sqrt(r2 + dr2 + 2*r*dr*np.cos(dphi))
        next_angle = np.arccos((r2 + next_r**2 - dr2)/2/r/next_r) + self.varphi
        # next_angle = self.varphi + dphi  
        return Position(next_r, next_angle)

    
    def generate_channel_gain(self, ts_t, location): 
        '''
        generate channel gain in dB 
        '''
        # generate the location 
        initial_loc = Location(location[0],location[1])
        mov_angle = location[2]
        velocity = location[3]

class User: 
    def __init__(self, init_param) -> None:
        self.loc = []
        self.loc.append(Position(init_param[0], init_param[1]))
        self.phi = init_param[2]
        self.velocity = init_param[3]
        self.channel_gain_dB = []
        self.channal_gain_wo_fading_dB = []
        pass

        mu, sigma = velocity*delta, 1/3*velocity*delta
        dr = rng.normal(mu, sigma, (no_slots)) # different location for each time slot 
        dvarphi_arr = []

        for i_t in range(1, no_slots): 
            curr_pos = self.loc[-1]

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
        h_tidle = rng.normal(mu_gain, sigma_gain, size=(no_slots)) # dB

        # calculate the channel gain 
        # channel_gain = (p_LOS + xi*(1 - p_LOS))*h_tidle*g0/((H_uav**2 + self.loc**2)**(gamma/2))
        distance = [(H_uav**2 + (position.r**2))**(gamma/2) for position in self.loc]
        gain_wo_fading_dB = todB(p_LOS + xi*(1 - p_LOS)) + g0 - todB(distance)
        gain_w_fading_dB = gain_wo_fading_dB + h_tidle

        self.channel_gain_dB = gain_w_fading_dB
        self.channal_gain_wo_fading_dB = gain_wo_fading_dB
        # x = [i for i in range(T)]

        # plt.figure(0)
        # plt.plot(x, gain_w_fading_dB, "-", x, gain_wo_fading_dB, '--')
        # plt.figure(1)
        # self.plot_location()
        # plt.show()
        print('finish showing!')

        return gain_w_fading_dB
    
    def plot_location(self): 
        
        xt = np.zeros((no_slots))
        yt = np.zeros((no_slots))
        for it, loc in enumerate(self.loc): 
            xt[it], yt[it] = loc.get_decart_pos() 
        print('start plotting') 
        plt.plot(xt[0], yt[0], 'o')
        plt.plot(xt, yt, '-', linewidth=1)

    def plot_channel_gain(self): 
        x = [t for t in range (no_slots)]
        plt.plot(x, self.gain_dB, '-', x, self.channal_gain_wo_fading_dB, '--', linewidth=0.8)

    def optimize_frequency(self, ts):  
        '''
        optimize the CPU frequency for local computation in time slot ts  
        '''
        Qt = self.Q[ts]
        f_hat = np.minimum(f_i_max, Qt*F/delta)
        f_i_L = np.minimum(np.sqrt(Qt/(3*kappa*V*F)), f_hat)
        
        energy = kappa * f_i_L**3 * delta
        at = np.round(f_i_L*delta/F)
        obj_val = V*energy - Qt*at 
        
        return at, energy, obj_val

    def update_queue(self, t, a_t, b_t, A_t):
        '''
        update the local queue in time slot t
        In: local volume at t a_t 
            offloaded volume at t b_t 
            arrival volume A_t
        ''' 
        self.Q[t+1] = np.minimum(0, self.Q[t] - a_t - b_t) + A_t

class Location: 
    def __init__(self, r, varphi):
        self.r = r
        self.varphi = varphi


if __name__ == '__main__':
    init_location = [[110, np.pi/4, np.pi, 1.5], 
        [110, np.pi*3/4, 0, 0],
        [10, 0, np.pi*3/4, 0.9],
        [80, -3/4*np.pi, np.pi/6, 1.5],
        [110, -3/4*np.pi, np.pi/12, 1.5]] 
    # init_location = [
    # [10, 0, np.pi*3/4, 0.9]] 
    users = [User(iloc) for iloc in init_location]
    
    for iuser in users: 
        iuser.generate_channel_gain()
        # iuser.plot_location()
    
    plt.figure(0)
    plt.grid()
    for iuser in users: 
        iuser.plot_location()
    plt.figure(1)
    plt.grid()
    for iuser in users: 
        iuser.plot_channel_gain()

    plt.show()
    print('simulation finish!')