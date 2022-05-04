from cProfile import label
import numpy as np 
import matplotlib.pyplot as plt
from psutil import users 

from User2 import User
from system_params import *
from Plot_figure import * 

path = create_img_folder()


class Server: 
    def __init__(self, arrival_rate): 
        self.L = np.zeros((T, N))
        self.users = []
        self.c = np.zeros((T, N))
        self.b = np.zeros((T, N))
        self.ts = 0 
        self.init_user(arrival_rate)

    def update_queue(self, idx): 
        idx = self.ts 
        # UAV queue update 
        self.L[idx + 1, :] = self.L[idx, :] - self.c[idx, :] + self.b[idx, :]
        # user queue update 
        Q = [user.update_queue(idx) for user in self.users]
        print('Function update_queue end!')
        

    def init_user(self, lambda_param): 
        self.users = [User(arrival_rate=lambda_param) for iuser in range(N)]
        self.plot()
        print('Finish')

    def plot(self):
        fig = plt.figure() 
        plt.grid()
        for (iuser, user) in enumerate(self.users): 
            plt.plot(range(T), user.data_A, label='user[{}]'.format(iuser))
            plt.xlabel('Time frame')
            plt.ylabel('Arrival task (packets)')
            plt.legend()

        plt.savefig(path + 'dataA.png'.format(iuser))
        plt.show()

    def opt_offloading_volume(self, mode, opt = 'ebw'):
        # input
        its = self.ts 
        idx1 = np.where(mode == 1)[0]

        # update local and remote queue of offloading user 
        Qt = np.array([user.Q[its, 0] for user in self.users])[idx1]
        Lt = self.L[its, idx1]
        gain = np.array([user.gain[its, 0] for user in self.users])[idx1]

        # Qt1 = Qt[idx1]
        # Lt1 = Lt[idx1]


        # optimize bandwidth 
        if opt == 'ebw': 
            bwidth = self.opt_bwidth_ebw(self, mode=mode)[idx1]
        else:
            bwidth = self.opt_bwidth_lawbert_w(mode=mode)[idx1]

        # for offloading users
        b_hat = np.round(bw*delta/R*np.log2(h*p_i_max/N0/bw) for (h, bw) in zip(gain, bwidth))

        b_hat = np.minimum(Qt, b_hat)
        


    
    def opt_bwidth_ebw(self, mode): 
        idx1 = np.where(mode == 1)[0]
        no_offloading_users = len(idx1)
        bwidth = np.zeros((N, 1))
        bwidth[idx1] = W/no_offloading_users

        return bwidth

    def opt_bwidth_lawbert_w(self, mode): 
        
        bwidth = np.zeros((N, 1))
        return bwidth
         





    
    def running(self): 
        for idx in range(T): 
            self.update_queue(idx)
        
        
        
        


    


s = Server(100)
s.running()
print('Finish!')