from cProfile import label
import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt 

from system_params import * 

def create_img_folder(): 
    # path = f'./V={V},dth={d_th},lambda={lambda_param},T={T},W={W},scale_delay={scale_delay}, psi = {psi}/'
    path = "{}/img/V ={:.2e},dth={:},lambda={:},T={:},W={:.2e},scale_delay={:}, psi = {:}, f_u ={:.2e}/".format(os.getcwd(),
     V, d_th, lambda_param, T, W, scale_delay, psi, f_u_max)
    os.makedirs(path, exist_ok=True)
    print(f"Directory {os.getcwd()}")
    return path

def plot_delay():
    csv_name = "V1.csv"
    # dth_arr = [1.5, 2.0, 3.0, 3.5, 4, 5, 8]
    # dth_arr = [2, 3, 4, 5, 6, 7, 8, 9]
    # dth_arr = [3, 4, 7, 9, 10, 15, 20, 25, 40]
    # xlabelstr = 'Delay threshold'

    dth_arr = [1e2, 1e3, 1e4, 1e5, 0.9*1e6, 1e6]
    xscale = 'log'
    xlabelstr = 'Lyapunov control parameter, V' 
    
    delay = np.zeros(len(dth_arr))
    user_energy = np.zeros(len(dth_arr))
    uav_energy = np.zeros(len(dth_arr))
    weighted_energy = np.zeros(len(dth_arr))
    weighted_energy2 = np.zeros(len(dth_arr))
    user_queue, uav_queue = np.zeros(len(dth_arr)), np.zeros(len(dth_arr))
    localA, offloadB, remoteC = np.zeros(len(dth_arr)), np.zeros(len(dth_arr)), np.zeros(len(dth_arr))
    


    for idx, dth in enumerate(dth_arr):
        path = "{}/img/V ={:.2e},dth={:},lambda={:},T={:},W={:.2e},scale_delay={:}, psi = {:}, f_u ={:.2e}/".format(os.getcwd(),
    dth, d_th, lambda_param, T, W, scale_delay, psi, f_u_max)
        
        file = path + csv_name
        data = pd.read_csv(file)
        delay[idx] = np.mean(data.delay)
        user_energy[idx] = np.mean(data.energy_user)*1000/delta
        uav_energy[idx] = np.mean(data.energy_uav)*1000/delta
        weighted_energy2[idx] = np.mean(data.energy_user + psi* data.energy_uav) * 1000/delta
        user_queue[idx] = np.mean(data.local_queue)
        uav_queue[idx] = np.mean(data.uav_queue)
        localA[idx] = np.mean(data.local_a)
        offloadB[idx] = np.mean(data.off_b)
        remoteC[idx] = np.mean(data.remote_c)


        # weighted_energy[idx] = np.mean(data.aweightedE)*1000/delta
        
    plt.plot(dth_arr, weighted_energy2, '-o', label='Weighted power')
    plt.plot(dth_arr, psi*uav_energy, '-o', label='UAV power')
    plt.plot(dth_arr, user_energy, '-o', label='User power')

    plt.xlabel(xlabelstr)
    plt.xscale(xscale)
    plt.ylabel('Power Consumption (mW)')
    plt.grid()
    plt.legend()
    plt.savefig('./img/' + xlabelstr+'_vs_power.png')
    plt.show()


    # plt.plot(dth_arr, dth_arr - delay, '-o', label = "Delta delay (TS)")
    plt.plot(dth_arr, delay, '-o', label = "Delay (TS)")
    
    plt.xscale(xscale)
    plt.xlabel(xlabelstr)
    plt.ylabel('Delay (TS)')
    plt.grid()
    plt.legend()
    plt.savefig('./img/' + xlabelstr + '_vs_dth.png')
    plt.show()

    plt.plot(dth_arr, offloadB, '-o', label = "Offloading packets")
    plt.plot(dth_arr, localA, '-o', label = "Local computation packets")
    plt.plot(dth_arr, remoteC, '-.', label = "UAV computation packets")

    plt.xscale(xscale)
    plt.xlabel(xlabel=xlabelstr)
    plt.ylabel('Computation and offload volume (TS)')
    plt.grid()
    plt.legend()
    plt.savefig('./img/' + xlabelstr + '_vs_abc.png')
    plt.show()

    plt.plot(dth_arr, uav_queue, '-o', label = "UAV queue")
    plt.plot(dth_arr, user_queue, '-o', label = "User queue")
    

    plt.xlabel(xlabel=xlabelstr)
    plt.xscale(xscale)
    plt.ylabel('Queue length (packets)')
    plt.grid()
    plt.legend()
    plt.savefig('./img/' + xlabelstr+'_vs_queue.png')
    plt.show()


# plot_delay()
