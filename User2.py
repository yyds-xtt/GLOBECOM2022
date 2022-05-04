import numpy as np
import matplotlib.pyplot as plt 

rng = np.random.default_rng()

from system_params import * 
import channelModel2

class User(): 
    def __init__(self, 
        arrival_rate
        ):
        self.loc = np.zeros((T, 1))
        self.gain = np.zeros((T, 1)) 
        self.energy = np.zeros((T, 1))
        self.Q = np.zeros((T, 1))
        # self.data_A = np.round(rng.uniform(low=0, high=2*arrival_rate, size=(T, 1)))
        self.data_A = np.round(rng.poisson(lam=arrival_rate, size=(T, 1)))
        self.a = np.zeros((T, 1))
        self.b = np.zeros((T, 1))
        self.gain = np.zeros((T, 1))
        self.idx = 0
        
    def update_queue(self, iidx): 
        # iidx = self.idx 
        if iidx > 0: 
            self.Q[iidx, 0] = self.Q[iidx - 1, 0] - self.a[iidx - 1, 0] - self.b[iidx - 1, 0] + self.data_A[iidx - 1, 0]
        return self.Q[iidx, 0]

    def next_slot(self): 
        # self.update_queue()
        # self.update_location()
        self.update_gain()
        self.idx += 1 
        pass 

    def update_gain(self): 
        self.gain[self.idx, 0] = channelModel2.channel_model()

    def optimize_freq(self): 
        Qt = self.Q[self.idx]
        f_hat = np.minimum(f_i_max, Qt*F/delta)
        ft = np.minimum(np.sqrt(Qt/3/F/V/kappa), f_hat)

        self.a[self.idx] = np.round(ft*delta/F)
        self.energy[self.idx] = kappa * (ft**3) * delta
        obj_value = V * self.energy[self.idx] - Qt*self.a[self.idx]

        return obj_value
    
    def plot_gain(self): 
        plt.plot(np.arange(T), self.gain)
        plt.xlim([0, T])
        plt.grid()
        plt.show()

    def plot_arrival(self): 
        plt.plot(np.arange(T), self.data_A)
        plt.xlim([0, T])
        plt.grid()
        plt.show()

# a = User(100)
# for i in np.arange(T):  
#     # a.optimize_freq()
#     a.next_slot()
#     print(i)
    # a.update_gain()

# a.plot_gain()
# a.plot_arrival()
# print('simulation finish!')