import numpy as np 
import matplotlib.pyplot as plt

rng = np.random.default_rng()

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
        self.users_energy = np.zeros((T, N))
        self.uav_energy = np.zeros((T, N))
        self.ts = 0 
        self.init_user(arrival_rate)

    def update_queue(self, idx): 
        idx = self.ts 
        # UAV queue update 
        self.L[idx + 1, :] = self.L[idx, :] - self.c[idx, :] + self.b[idx, :]
        # user queue update 
        # Q = [user.update_queue() for user in self.users]
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

        '''
        return number of offloading packets 
        '''
        # input
        its = self.ts 
        idx1 = np.where(mode == 1)[0]

        # update local and remote queue of offloading user 
        Qt = np.array([user.Q[its, 0] for user in self.users])[idx1]
        Lt = self.L[its, idx1]
        gain = np.array([user.gain[its, 0] for user in self.users])[idx1]

        # Qt1 = Qt[idx1]
        # Lt1 = Lt[idx1]
        users_energy = np.zeros((N))
        uav_energy = np.zeros((N))


        # optimize bandwidth 
        if opt == 'ebw': 
            bwidth = self.opt_bwidth_ebw(mode=mode)[0][idx1]
        else:
            bwidth = self.opt_bwidth_lawbert_w(mode=mode)[0][idx1]

        # for offloading users
        b_hat = np.round(bwidth*delta/R*np.log2(gain*p_i_max/N0/bwidth))

        b_hat = np.minimum(Qt, b_hat)

        bt = np.zeros_like(gain)
        iidx = Qt > Lt 
        tmp_bt = np.round(bwidth[iidx]*delta/R * np.log2(np.multiply(gain[iidx], Qt[iidx] - Lt[iidx]))/(V * N0 * R * np.log(2)))

        bt[iidx] = np.maximum(0, tmp_bt)

        bt_arr = np.zeros((N))
        bt_arr[idx1] = bt

        
        users_energy[idx1] = (N0*bwidth*delta/gain)*(2**(bt*R/bwidth/delta) - 1)
        fvalue = np.sum(-bt*(Qt - Lt) + V * users_energy[idx1])     
        result = {"fvalue": fvalue,
                "users_energy": users_energy,
                "off_bt": bt_arr
                }

        return result

    def opt_uav_freq(self, mode): 
        # input
        its = self.ts 
        idx1 = np.where(mode == 1)[0]
        f_uav = np.zeros_like(idx1)

        # update local and remote queue of offloading user 
        Lt = self.L[its, idx1]
        f_hat = np.minimum(f_u_max, Lt*F/delta)
        f_uav = np.sqrt(Lt/3/F/V/kappa/psi)
        f_uav = np.minimum(f_uav, f_hat)
        # self.uav_energy[its, idx1] = kappa * (f_uav**3) * delta
        uav_energy = np.zeros((N))
        uav_energy[idx1] = kappa * (f_uav**3) * delta
        fvalue =np.sum(-Lt * f_uav * delta/F + V * psi * uav_energy[idx1])
        result = {'fvalue': fvalue, 
                'uav_energy': uav_energy
                }

        return result
    
    def opt_bwidth_ebw(self, mode): 
        idx1 = np.where(mode == 1)[0]
        no_offloading_users = len(idx1)
        bwidth = np.zeros((1, N))
        bwidth[:, idx1] = W/no_offloading_users

        return bwidth

    def opt_bwidth_lawbert_w(self, mode): 
        
        bwidth = np.zeros((N, 1))
        return bwidth
         





    
    def running(self): 
        T = 3 
        for idx in range(T): 
            self.update_queue(idx)
            for user in self.users: 
                user.next_slot()
        mode = rng.integers(low=0, high=2, size=N)

        self.opt_offloading_volume(mode)
        
        
        
        
        


    


s = Server(100)
s.running()
print('Finish!')