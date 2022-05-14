#  #################################################################
#
#  This file contains the main code of LyDROO.
#
#  References:
#  [1] Suzhi Bi, Liang Huang, Hui Wang, and Ying-Jun Angela Zhang, "Lyapunov-guided Deep Reinforcement Learning for Stable Online Computation Offloading in Mobile-Edge Computing Networks," IEEE Transactions on Wireless Communications, 2021, doi:10.1109/TWC.2021.3085319.
#  [2] Liang Huang, Suzhi Bi, and Ying-Jun Angela Zhang, "Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks," in IEEE Transactions on Mobile Computing, vol. 19, no. 11, pp. 2581-2593, November 2020.
#  [3] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” IEEE Trans. Wireless Commun., vol. 17, no. 6, pp. 4177-4190, Jun. 2018.
#
# version 1.0 -- July 2020. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################


import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import pandas as pd 

from pandas import DataFrame as df 

from sklearn.preprocessing import MinMaxScaler


# for tensorflow2
from memoryTF2conv import MemoryDNN
from Plot_figure import * 
# from optimization import bisection
from ResourceAllocation import Algo1_NUM

from system_params import d_th, scale_delay, V
import channelModel2
import gen_arrival_tasks
from User import *
from utils import * 

import time

import os 

def preprocessing(data_in):
    # create scaler 
    scaler = MinMaxScaler()
    data = np.reshape(data_in, (-1, 1))
    # fit scaler on data 
    scaler.fit(data)
    normalized = scaler.transform(data)
    normalized = normalized.reshape(1, -1)
    return normalized 

no_input_cnn = 7

np.set_printoptions(precision=2)

def gen_actions_bf(no_users = 5): 
    import itertools
    actions = np.array(list(itertools.product([0, 1], repeat=no_users))) # (32, 5)

    return actions


if comparison_flag == False: 
    channelModel2.gen_channel_gain()
    gen_arrival_tasks.gen_arrival_tasks()

if __name__ == "__main__":
    '''
        LyDROO algorithm composed of four steps:
            1) 'Actor module'
            2) 'Critic module'
            3) 'Policy update module'
            4) ‘Queueing module’ of
    '''

    path = create_img_folder()
    # V = 10**V

    K = N                   # initialize K = N


    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,T,K,decoder_mode, Memory, Delta))

    init_location = [[110, np.pi/4, np.pi, 1.5], 
        [110, np.pi*3/4, 0, 0],
        [10, 0, np.pi*3/4, 0.9],
        [80, -3/4*np.pi, np.pi/6, 1.5],
        [110, -3/4*np.pi, np.pi/12, 1.5],
        [110, np.pi/4, np.pi, 1.5], 
        [110, np.pi*3/4, 0, 0],
        [10, 0, np.pi*3/4, 0.9],
        [80, -3/4*np.pi, np.pi/6, 1.5],
        [110, -3/4*np.pi, np.pi/12, 1.5]]

    users = [User(iloc) for iloc in init_location]
    
    # for iuser in users: 
    #     iuser.generate_channel_gain()
    h = np.zeros((T, N))
    
    mem = MemoryDNN(net = [N*no_input_cnn, 256, 128, N],
                    learning_rate = 0.01,
                    training_interval=20,
                    batch_size=128,
                    memory_size=Memory)

    start_time=time.time()
    mode_his = np.zeros((T, N))# store the offloading mode
    k_idx_his = [] # store the index of optimal offloading actor
    Q = np.zeros((T,N)) # local queue in tasks
    L = np.zeros((T,N)) # UAV queue in tasks
    D = np.zeros((T,N)) # delay in time slots

    Obj = np.zeros((T)) # objective values after solving problem (26)
    energy = np.zeros((T,N)) # energy consumption
    energy_uav = np.zeros((T, N))
    rate = np.zeros((T,N)) # achieved computation rate
    d_t = np.zeros((T, N)) 
    
    a = np.zeros((T, N)) # number of local computation tasks 
    b = np.zeros((T, N)) # number of offloading tasks 
    c = np.zeros((T, N))  # number of remote computation tasks
    delay = np.zeros((T, N)) # estimated delay 

    drift_Q = np.zeros((T))
    drift_L = np.zeros((T))
    drift_D = np.zeros((T))
    weighted_energy = np.zeros((T))

    # idx = 0
    # data A load from system_params 
    Q[0,:] = dataA[0,:] # save current data queue

    for i in range(1, T):

        if i % (T//10) == 0:
            print("%0.1f"%(i/T))
        if i> 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(np.array(k_idx_his[-Delta:-1])%K) +1
            else:
                max_k = k_idx_his[-1] +1
            K = min(max_k +1, N)
            # K = 40

        i_idx = i
        K = 100 
            
        h = dB(channel_gain[i, :])

        # 4) ‘Queueing module’ of LyDROO
        
        # update queues
        Q[i_idx,:] = np.maximum(0, Q[i_idx-1,:] - a[i_idx-1,:] - b[i_idx-1,:]) + dataA[i_idx-1,:] # current data queue
        L[i_idx,:] = np.maximum(0, L[i_idx-1,:] - c[i_idx-1,:]) + b[i_idx-1,:]
        D[i_idx,:] = np.maximum(0, D[i_idx-1,:] + scale_delay*delay[i_idx-1, :] - scale_delay*d_th)

        drift_Q[i_idx] = 1/2 *np.sum((Q[i_idx,:] - Q[i_idx - 1,:])**2)
        drift_L[i_idx] = 1/2 *np.sum((L[i_idx,:] - L[i_idx - 1,:])**2)
        drift_D[i_idx] = 1/2 *np.sum((D[i_idx,:] - D[i_idx - 1,:])**2)

        # scale Q and Y to 1
        # h_norm = np.linalg.norm(h)
        
        # q_norm = np.linalg.norm(Q[i_idx,:])
        # l_norm = np.linalg.norm(L[i_idx,:])
        # d_norm = np.linalg.norm(D[i_idx,:])

        # if q_norm == 0: 
        #     q_norm = 1 
        # if l_norm == 0: 
        #     l_norm = 1 
        # if d_norm == 0: 
        #     d_norm = 1 
        # nn_input =np.vstack((h, Q[i_idx,:]/q_norm, L[i_idx,:]/l_norm,D[i_idx, :]/d_norm)).transpose().flatten()
        b_idx = np.maximum(0, i_idx - 30)
        h_normalized = preprocessing(h*CHFACT)
        Q_normalized = preprocessing(Q[i_idx,:])
        L_normalized = preprocessing(L[i_idx,:])
        D_normalized = preprocessing(D[i_idx, :])
        Q_ava_normalized = preprocessing(np.mean(Q[b_idx:i_idx+1, :], axis=0))
        L_ava_normalized = preprocessing(np.mean(L[b_idx:i_idx+1, :], axis=0))
        b_ava_normalized = preprocessing(np.mean(b[b_idx:i_idx+1, :], axis=0))
        nn_input =np.vstack((h_normalized, Q_normalized, L_normalized, D_normalized, Q_ava_normalized, L_ava_normalized, b_ava_normalized)).transpose().flatten()



        # 1) 'Actor module' of LyDROO
        # generate a batch of actions
        if opt_mode == 'bf': 
            m_list = gen_actions_bf(no_users=N).copy()
        elif opt_mode == 'LYDROO': 
            m_list = mem.decode(nn_input, K, decoder_mode)

        r_list = [] # all results of candidate offloading modes
        v_list = [] # the objective values of candidate offloading modes
        delay_list = []
        for m in m_list:
            # 2) 'Critic module' of LyDROO
            # allocate resource for all generated offloading modes saved in m_list
            r_list.append(Algo1_NUM(m,h,Q[i_idx,:],L[i_idx,:],V))
            # v_list.append(r_list[-1][0])
            f_val = r_list[-1][0]
            
            tmp, a_i, b_i, c_i, energy_i, energy_uav_i = r_list[-1]

            # # estimate the current value delay
            d_i_t = np.zeros(N)
            Q_i_t = np.zeros(N)
            L_i_t = np.zeros(N)
            b_i_t = np.zeros(N)
            # avarage local queue 

            b_idx = np.maximum(0, i_idx + 1 - window_size) 

            
            # if mode == 'test':
            Q_i_t = find_mean_queue(Q[b_idx:i_idx+1, :], a_i + b_i, dataA[i_idx,:]) 
            L_i_t = find_mean_queue(L[b_idx:i_idx+1, :], c_i, b_i)  # average uav queue          

            d_i_t = (Q_i_t +L_i_t)/lambda_param
                
            f_val = f_val + np.sum(1/2*(scale_delay*d_i_t)**2 + scale_delay*d_i_t*(D[i_idx,:] - scale_delay*d_th)) # update the objective function
            
            v_list.append(f_val)
            delay_list.append(d_i_t)
            
        # record the index of largest reward
        k_idx_his.append(np.argmin(v_list))

        # 3) 'Policy update module' of LyDROO
        # encode the mode with largest reward
        mem.encode(nn_input, m_list[k_idx_his[-1]])
        mode_his[i_idx, :] = m_list[k_idx_his[-1]]

        # store max result
        # Obj[i_idx],rate[i_idx,:],energy[i_idx,:]  = r_list[k_idx_his[-1]]
        Obj[i_idx] = v_list[k_idx_his[-1]]

        ###############################
        delay[i_idx] = delay_list[k_idx_his[-1]]

        tmp, a[i_idx,:],b[i_idx,:], c[i_idx,:], energy[i_idx, :], energy_uav[i_idx, :] = r_list[k_idx_his[-1]]


        # drifted energy 
        weighted_energy[i_idx] = (np.sum(energy[i_idx, :] + energy_uav[i_idx, :]*psi))
        is_debug_mode = True 
        if is_debug_mode and i%100 == 0: 
            print(f'local computation: a_i =', a[i_idx,:])
            print(f'offloading volume: b_i =', b[i_idx,:])
            print(f'remote computation: c_i =', c[i_idx,:])
            print(f'remote computation: energy_i =', energy[i_idx,:])
            print(f'delay: delay_i =', delay[i_idx])
            print(f'virtualqueue_i =', D[i_idx,:])
            print(f'local queue: queue_i =', Q[i_idx,:])

            print(f'fvalue = {v_list[k_idx_his[-1]]}')
    
    plot_kpi_users(A=dataA, a=a, b=b, Q=Q, name=path)
    plot_kpi_uav(L=L, b=b, c=c, name=path)
    plot_kpi_avr(Q, L, D, energy, delay, energy_uav, path)
    plot_kpi_drift(drift_Q, drift_L, drift_D, weighted_energy, path)



    total_time=time.time()-start_time
    print(f'total time = {total_time}')
    mem.plot_cost(path_name=path+'TraningLoss')


    
    print(f"Mean drift local queue: {np.mean(drift_Q)}")
    print(f"Mean drift uav queue: {np.mean(drift_L)}")
    print(f"Mean drift delay: {np.mean(drift_D)}")
    print(f"Mean drift energy: {np.mean(weighted_energy)*V}")

    print('Average time per channel:%s'%(total_time/T))

    # save all data
    aQ = np.mean(Q, axis=1)
    aL = np.mean(L, axis=1)
    aE_i = np.mean(energy, axis=1)
    aE_u = np.mean(energy_uav, axis=1)
    aweightedE = weighted_energy/N
    adelay = np.mean(delay, axis=1)
    aOffloadingb = np.mean(b, axis=1)
    aLocala = np.mean(a, axis=1)
    aUAVc = np.mean(c, axis=1)
    
    # sio.savemat('./result_%d.mat'%N, {'input_h': channel/CHFACT,'data_arrival':dataA,'local_queue':Q,'uav_queue':L,'off_mode':mode_his,'energy_consumption':energy,'delay':delay,'objective':Obj})
    df = pd.DataFrame({'local_queue':aQ,'uav_queue':aL,
                        'energy_user':aE_i,'energy_uav':aE_u, 
                        'delay':adelay, 'weightedE':aweightedE, 
                        'off_b': aOffloadingb, 'local_a': aLocala, 'remote_c': aUAVc, 
                        'time': total_time})
    name= path + 'V1.csv'
    df.to_csv(name, index=False)
    print('completed!')
