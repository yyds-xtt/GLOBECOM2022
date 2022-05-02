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

def plot_scale_factor(): 
    csv_name = "V1.csv"
    # path = create_img_folder()
    print('create img foler')
    # dth_arr = [2.5, 3, 4, 5, 6, 7, 8, 9]


    scale_delay_arr = np.asarray([0.01, 0.1, 1.0, 10, 100, 1000])*1e2
    delay_arr = np.zeros_like(scale_delay_arr)
    energy_arr = np.zeros_like(scale_delay_arr)
    


    for idx, scale in enumerate(scale_delay_arr):
        path = "{}/img/V ={:.2e},dth={:},lambda={:},T={:},W={:.2e},scale_delay={:}, psi = {:}, f_u ={:.2e}/".format(os.getcwd(),
    V, d_th, lambda_param, T, W, scale, psi, f_u_max)
        
        file = path + csv_name
        data = pd.read_csv(file)
        delay_arr[idx] = np.mean(data.delay)
        energy_arr[idx] = np.mean(data.weightedE)*1000
        
    plt.plot(scale_delay_arr, delay_arr, '--o')
    plt.ylabel('Delay (TS)')
    plt.xscale('log')
    plt.xlabel('Delay scale factor')
    plt.grid()
    plt.legend()
    plt.savefig('Delay_vs_scalefactor.png')
    plt.show()


    plt.plot(scale_delay_arr, energy_arr, '-o', label = "Weighed energy (mJ)")
    plt.ylabel('Energy comsumption')
    plt.xlabel('Delay scale factor')
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.savefig('Energy_vs_scalefactor.png')
    plt.show()

# plot_figure()

def plot_delay():
    csv_name = "V1.csv"
    dth_arr = [1.0, 2.5, 3, 4]
    delay = np.zeros_like(dth_arr)
    user_energy = np.zeros_like(dth_arr)
    uav_energy = np.zeros_like(dth_arr)
    weighted_energy = np.zeros_like(dth_arr)

    for idx, dthreshold in enumerate(dth_arr):
        path = "{}/img/V ={:.2e},dth={:},lambda={:},T={:},W={:.2e},scale_delay={:}, psi = {:}, f_u ={:.2e}/".format(os.getcwd(),
    V, dthreshold, lambda_param, T, W, scale_delay, psi, f_u_max)
        
        file = path + csv_name
        data = pd.read_csv(file)
        delay[idx] = np.mean(data.delay)
        user_energy[idx] = np.mean(data.energy_user)*1000
        uav_energy[idx] = np.mean(data.energy_uav)*1000
        weighted_energy[idx] = np.mean(data.weighted_energy)*1000
        
    plt.plot(dth_arr, weighted_energy, label='Weighted energy')
    plt.plot(dth_arr, uav_energy, label='UAV energy')
    plt.plot(dth_arr, user_energy, label='User energy')

    plt.ylabel('Energy Consumption (mJ)')
    plt.xlabel('Delay threshold (TS)')
    plt.grid()
    plt.legend()
    plt.savefig('energy_vs_dth.png')
    plt.show()


    plt.plot(dth_arr, delay, '-o', label = "Delay (mJ)")
    plt.ylabel('Delay')
    plt.xlabel('Delay threshold (TS)')
    plt.grid()
    plt.legend()
    plt.savefig('delay_vs_dth.png')
    plt.show()

plot_delay()
