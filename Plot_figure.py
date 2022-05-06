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
def plot_lyapunov(): 
    csv_name = "V1.csv"
    # path = create_img_folder()
    print('create img foler')
    V_arr = np.array([1e4, 1e5, 1e6])


    delay = np.zeros(len(V_arr))
    user_energy = np.zeros(len(V_arr))
    uav_energy = np.zeros(len(V_arr))
    user_queue, uav_queue = np.zeros_like(V_arr), np.zeros_like(V_arr)
    weighted_energy2 = np.zeros(len(V_arr))
    


    for idx, _V in enumerate(V_arr):
        path = "{}/img/V ={:.2e},dth={:},lambda={:},T={:},W={:.2e},scale_delay={:}, psi = {:}, f_u ={:.2e}/".format(os.getcwd(),
    _V, d_th, lambda_param, T, W, scale_delay, psi, f_u_max)
        
        file = path + csv_name
        data = pd.read_csv(file)
        delay[idx] = np.mean(data.delay)
        user_energy[idx] = np.mean(data.energy_user)*1000/delta
        uav_energy[idx] = np.mean(data.energy_uav)*1000/delta
        weighted_energy2[idx] = np.mean(data.energy_user + psi* data.energy_uav) * 1000/delta
        user_queue[idx] = np.mean(data.local_queue)
        uav_queue[idx] = np.mean(data.uav_queue)        
    plt.plot(V_arr, weighted_energy2, '--o')
    plt.ylabel('Weighted energy consumption')
    plt.xscale('log')
    plt.xlabel('Lyapunov parameter')
    plt.grid()
    plt.legend()
    plt.savefig('V_energy.png')
    plt.show()

    plt.plot(V_arr, delay, '--o')
    plt.ylabel('Delay')
    plt.xscale('log')
    plt.xlabel('Lyapunov parameter')
    plt.grid()
    plt.legend()
    plt.savefig('V_delay.png')
    plt.show()

    plt.plot(V_arr, user_queue, '--o', label = 'Average UE queue')
    plt.plot(V_arr, uav_queue, '-o',  label = 'Average UAV queue')
    plt.ylabel('Queue length')
    plt.xscale('log')
    plt.xlabel('Lyapunov parameter')
    plt.grid()
    plt.legend()
    plt.savefig('V_queue.png')
    plt.show()

def plot_delay():
    csv_name = "V1.csv"
    # dth_arr = [1.5, 2.0, 3.0, 3.5, 4, 5, 8]
    dth_arr = [2, 3, 4, 5, 6, 7, 8, 9]
    
    delay = np.zeros(len(dth_arr))
    user_energy = np.zeros(len(dth_arr))
    uav_energy = np.zeros(len(dth_arr))
    weighted_energy = np.zeros(len(dth_arr))
    weighted_energy2 = np.zeros(len(dth_arr))
    user_queue, uav_queue = np.zeros(len(dth_arr)), np.zeros(len(dth_arr))


    for idx, dthreshold in enumerate(dth_arr):
        path = "{}/img/V ={:.2e},dth={:},lambda={:},T={:},W={:.2e},scale_delay={:}, psi = {:}, f_u ={:.2e}/".format(os.getcwd(),
    V, dthreshold, lambda_param, T, W, scale_delay, psi, f_u_max)
        
        file = path + csv_name
        data = pd.read_csv(file)
        delay[idx] = np.mean(data.delay)
        user_energy[idx] = np.mean(data.energy_user)*1000/delta
        uav_energy[idx] = np.mean(data.energy_uav)*1000/delta
        weighted_energy2[idx] = np.mean(data.energy_user + psi* data.energy_uav) * 1000/delta
        user_queue[idx] = np.mean(data.local_queue)
        uav_queue[idx] = np.mean(data.uav_queue)
        # weighted_energy[idx] = np.mean(data.aweightedE)*1000/delta
        
    plt.plot(dth_arr, weighted_energy2, '-o', label='Weighted power')
    plt.plot(dth_arr, uav_energy, '-o', label='UAV power')
    plt.plot(dth_arr, user_energy, '-o', label='User power')
    plt.plot(dth_arr, q)

    plt.ylabel('Power Consumption (mW)')
    plt.xlabel('Delay threshold (TS)')
    plt.grid()
    plt.legend()
    plt.savefig('energy_vs_dth_V5.png')
    plt.show()


    plt.plot(dth_arr, delay, '-o', label = "Delay (TS)")
    plt.ylabel('Delay')
    plt.xlabel('Delay threshold (TS)')
    plt.grid()
    plt.legend()
    plt.savefig('delay_vs_dth_V5.png')
    plt.show()

# plot_delay()
# plot_scale_factor()
plot_lyapunov()
