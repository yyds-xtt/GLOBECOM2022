from cProfile import label
import numpy as np 
import matplotlib.pyplot as plt 

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
        self.idx = 0 
        self.init_user(arrival_rate)

    def update_queue(self, idx): 
        # idx = self.idx 
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
    
    def running(self): 
        for idx in range(T): 
            self.update_queue(idx)
        
        
        
        


    


s = Server(100)
s.running()
print('Finish!')