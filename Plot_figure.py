from cProfile import label
import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt
from system_params import * 

def create_img_folder(): 

    path = "{}/img/{}, V ={:.2e}, dth={:},lambda={:}, mode={}, window_size = {}/".format(os.getcwd(),
    opt_mode, V, d_th, lambda_param, mode, window_size)
    os.makedirs(path, exist_ok=True)
    print(f"Directory {os.getcwd()}")
    return path

def plot_kpi_users(A, a, b, Q, name):
    kpi_list = ['Arrival tasks', 'Local tasks', 'Offloaded tasks', 'Queue length']
    # dataA, local, uav, queue = user.dataA, user.a, user.b, user.Q
    # n_rows, n_cols = user.dataA.shape   #(T, 1)
    n_ts, n_users = A.shape
    for iuser in range(n_users):
        fig = plt.figure(iuser)
        plt.plot(np.arange(n_ts), A[:, iuser], label=kpi_list[0])
        plt.plot(np.arange(n_ts), a[:, iuser], label=kpi_list[1])
        plt.plot(np.arange(n_ts), b[:, iuser], label=kpi_list[2])
        plt.plot(np.arange(n_ts), Q[:, iuser], label=kpi_list[3])
        plt.title('user[{}]'.format(iuser))
        plt.xlabel('Time frames')
        plt.legend()
        plt.grid()
        plt.savefig(name+'_user[{}]'.format(iuser))
        # plt.show()
        plt.close()
        #

def plot_kpi_uav(L, b, c, name): 
    kpi_list = ['Arrival tasks', 'Computed tasks', 'UAV Queue length']
    # dataA, local, uav, queue = user.dataA, user.a, user.b, user.Q
    # n_rows, n_cols = user.dataA.shape   #(T, 1)
    n_ts, n_users = L.shape
    for iuser in range(n_users):
        fig = plt.figure(iuser)
        plt.plot(np.arange(n_ts), b[:, iuser], label=kpi_list[0])
        plt.plot(np.arange(n_ts), c[:, iuser], label=kpi_list[1])
        plt.plot(np.arange(n_ts), L[:, iuser], label=kpi_list[2])
        plt.xlabel('user[{}]'.format(iuser))
        plt.legend()
        plt.grid()
        plt.savefig(name+'_uav_user[{}]'.format(iuser))
        # plt.show()
        plt.close()
    
def plot_kpi_avr(Q, L, D, energy, delay, energy_uav, path): 
    rolling_intv = 10
    plot_rate(Q.sum(axis=1)/N, rolling_intv, 'User queue length', name=path+'UserQueue')
    plot_rate(L.sum(axis=1)/N, rolling_intv, 'UAV queue length', name=path+'UAVQueue')
    plot_rate(D.sum(axis=1)/N, rolling_intv, 'virtual queue', name=path+'DelayQueue')
    plot_rate(energy.sum(axis=1)/N/delta*1000, rolling_intv, 'Power consumption (mW)', name=path+'AvgPower')
    plot_rate(delay.sum(axis=1)/N, rolling_intv, 'Latency (TS)',name=path+'AvgDelay')
    plot_rate(np.sum(energy_uav, axis=1)/N/delta*1000, rolling_intv, 'UAV power consumption (mW)', name=path+'AvgPowerUAV')
    
def plot_kpi_drift(drift_Q, drift_L, drift_D, weighted_energy, path): 
    plot_drift(drift_Q, drift_L, drift_D, weighted_energy*V, name=path+"drift")
    plt.close()

def plot_delay():
    csv_name = "V1.csv"
    dth_arr = [6, 8, 10, 16, 20]
    # xscale = 'linear'

    xlabelstr = 'Delay threshold'
    # dth_arr = [1e3, 1.5*1e3, 3*1e3, 5*1e3, 7*1e3, 8*1e3, 9*1e3, 1e4, 2*1e4, 5*1e4, 1e5]
    # dth_arr = [5, 10, 15, 18, 20]
    # dth_arr = [1e3, 8*1e3, 2*1e4 ,1e5]

    # xscale = 'log'
    # xlabelstr = 'Lyapunov control parameter, V' 
    
    xscale = 'linear'
    # xlabelstr = 'Scale delay factor'
    
    delay = np.zeros(len(dth_arr))
    user_energy = np.zeros(len(dth_arr))
    uav_energy = np.zeros(len(dth_arr))
    weighted_energy = np.zeros(len(dth_arr))
    weighted_energy2 = np.zeros(len(dth_arr))
    user_queue, uav_queue = np.zeros(len(dth_arr)), np.zeros(len(dth_arr))
    localA, offloadB, remoteC = np.zeros(len(dth_arr)), np.zeros(len(dth_arr)), np.zeros(len(dth_arr))

    


    for idx, dth in enumerate(dth_arr):
        d_th = dth_arr[idx]
        # scale_delay = dth_arr[idx]
        # V = dth_arr[idx]

        path = "{}/img/V ={:.2e},dth={:},lambda={:},T={:},W={:.2e},scale_delay={:}, psi = {:}, f_u ={:.2e}, mode={}, window_size = {}/".format(os.getcwd(),
     V, d_th, lambda_param, T, W, scale_delay, psi, f_u_max, mode, window_size)
        
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
    plt.plot(dth_arr, user_energy, '-ob', label='User power')
    plt.plot(dth_arr, psi*uav_energy, '-or', label='Weighted UAV power')        
    plt.plot(dth_arr, weighted_energy2, '-ok', label='Weighted power')

    plt.xlabel(xlabelstr)
    plt.xticks(dth_arr)
    plt.xscale(xscale)
    plt.ylabel('Power consumption (mW)')
    plt.grid()
    plt.legend()
    plt.savefig('./img/' + xlabelstr+'_vs_power.png')
    plt.show()


    # plt.plot(dth_arr, dth_arr - delay, '-o', label = "Delta delay (TS)")
    plt.plot(dth_arr, delay, '-ob', label = "Delay")
    
    plt.xscale(xscale)
    plt.xticks(dth_arr)

    plt.xlabel(xlabelstr)
    plt.ylabel('Delay (TS)')
    plt.grid()
    plt.legend()
    plt.savefig('./img/' + xlabelstr + '_vs_dth.png')
    plt.show()

    plt.plot(dth_arr, localA, '-ob', label = "Local computation packets")
    plt.plot(dth_arr, offloadB, '-or', label = "Offloading packets")
    plt.plot(dth_arr, remoteC, '-.ok', label = "UAV computation packets")

    plt.xscale(xscale)
    plt.xticks(dth_arr)
    plt.xlabel(xlabel=xlabelstr)
    plt.ylabel('Computation and offload volume (packets)')
    plt.grid()
    plt.legend()
    plt.savefig('./img/' + xlabelstr + '_vs_abc.png')
    plt.show()

    plt.plot(dth_arr, user_queue, '-ob', label = "User queue")
    plt.plot(dth_arr, uav_queue, '-or', label = "UAV queue")
    plt.xlabel(xlabel=xlabelstr)
    plt.xticks(dth_arr)

    plt.xscale(xscale)
    plt.ylabel('Queue length (packets)')
    plt.grid()
    plt.legend()
    plt.savefig('./img/' + xlabelstr+'_vs_queue.png')
    # plt.show()

# plot_delay() 


def plot_drift(Q, L, D, E, name, rolling_intv=50): 
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    # mpl.style.use('seaborn')
    # fig, ax = plt.subplots(figsize=(8,8))
    plt.grid()
    plt.xlim([1, len(Q)+1])

    rate_array = np.asarray(Q)
    df = pd.DataFrame(Q)
    # plt.plot(np.arange(T), Q, np.arange(T), L, np.arange(T), D, np.arange(T), E)
    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))

    df = pd.DataFrame(L)
    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))
    df = pd.DataFrame(D)
    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))
    df = pd.DataFrame(E)
    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))



    plt.xlabel('Time Frames')
    plt.legend(['Q','L', 'D', 'E'])
    plt.savefig(name)
    plt.show()


def plot_rate( rate_his, rolling_intv = 10, ylabel='Normalized Computation Rate', name='Average queue length'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    # mpl.style.use('seaborn')
    # fig, ax = plt.subplots(figsize=(15,8))
    plt.grid()
    plt.plot(np.arange(len(rate_array))+1, rate_array)
    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = 'b', alpha = 0.2)
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.savefig(name)
    plt.show()
    plt.close()


def plot_comparison():
    csv_name = "V1.csv"
    no_users = [8, 10, 12]
    no_rows, no_cols = len(opt_mode_arr), len(no_users)

    delay, time, energy = np.zeros((no_rows, no_cols)), np.zeros((no_rows, no_cols)), np.zeros((no_rows, no_cols))
# load data 
    ac_data = np.zeros((3, no_rows, no_cols))
    for imode in np.arange(no_rows): 
        for iuser in np.arange(no_cols): 
            
            opt_mode = opt_mode_arr[imode]
            nuser = no_users[iuser]
            print("mode = {}, user = {}".format(opt_mode, nuser))
            path = "{}/img/{}, V ={:.2e}, dth={:},T={},lambda={:},W={:.2e},scale_delay={:}, psi = {:}, f_u ={:.2e}, mode={}, window_size = {}, no_users={}/".format(os.getcwd(),
            opt_mode, V, d_th, T, lambda_param, W, scale_delay, psi, f_u_max, mode, window_size, nuser)
            
            name = path + csv_name

            data = pd.read_csv(name)
            delay[imode, iuser] = np.mean(data.delay)
            energy[imode, iuser] = np.mean(data.energy_user + psi* data.energy_uav) * 1000/delta
            time[imode, iuser] = np.mean(data.time)
    ac_data[0, :, :] = time.copy()
    ac_data[1, :, :] = energy.copy()
    ac_data[2, :, :] = delay.copy()

# plot 
#   
    y_label_arr = ['Computation time', 'Energy Consumption', 'Delay']
    name_arr = ['Time', 'Energy',  'Delay']
    title = [y_label + ' vs. #UEs' for y_label in y_label_arr]

    for idx in range(3):
        data = ac_data[idx, :, :].copy()
        x = np.arange(len(no_users))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, data[0, :], width, label=opt_mode_arr[0])
        rects2 = ax.bar(x + width/2, data[1, :], width, label=opt_mode_arr[1])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(y_label_arr[idx])
        ax.set_title(title[idx])
        ax.set_xticks(x)
        ax.set_xticklabels(no_users)
        
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.2f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        
        ax.legend()
        plt.savefig('{}/img/cmq_{}_UEs.png'.format(os.getcwd(), name_arr[idx]))
        plt.show()
        plt.close()

    print('load completed')

# plot_comparison()
