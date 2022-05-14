import numpy as np 
import pandas as pd 
from system_params import N, T, lambda_param

def gen_arrival_tasks(): 
    arrival_lambda = lambda_param*np.ones((N)) # 1.5 Mbps per user
    dataA = np.round(np.random.poisson(arrival_lambda, size=(T, N)))
    # df = pd.DataFrame({
    #     'dataA': dataA
    # })


    # df.to_csv('dataA.csv', index=False)
    np.save('dataA', dataA)
    return dataA

gen_arrival_tasks()

dataA = np.load('dataA.npy')
print("load completed")
