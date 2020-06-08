import math
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel



def USA_data(directory ):
    signals =  pd.read_csv( directory + 'Usa_temp.csv')
    if "Unnamed: 0" in signals.columns:
        signals.drop(columns="Unnamed: 0", inplace = True)
    A = np.load( directory + 'Adjk10_07-13.npy')     

    return signals, A


def Seattle_data(directory , binary=False):
    """
    Seattle_data:  https://github.com/zhiyongc/Graph_Convolutional_LSTM/blob/master/Code_V2/HGC_LSTM%20%26%20Experiments.ipynb

    Args:
        directory ([type]): [description]
        binary (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    speed_matrix =  pd.read_pickle( directory + 'speed_matrix_2015',)
    A = np.load( directory + 'Loop_Seattle_2015_A.npy')

    if not binary:
        cor = rbf_kernel(speed_matrix[:1000].T/10)  
        A = cor*(A)       
        e, V = np.linalg.eigh(A)
        A/=np.max(e)
        A = A-np.diag(A.diagonal())

    FFR_5min = np.load( directory + 'Loop_Seattle_2015_reachability_free_flow_5min.npy')
    FFR_10min = np.load( directory + 'Loop_Seattle_2015_reachability_free_flow_10min.npy')
    FFR_15min = np.load( directory + 'Loop_Seattle_2015_reachability_free_flow_15min.npy')
    FFR_20min = np.load( directory + 'Loop_Seattle_2015_reachability_free_flow_20min.npy')
    FFR_25min = np.load( directory + 'Loop_Seattle_2015_reachability_free_flow_25min.npy')
    FFR = [FFR_5min, FFR_10min, FFR_15min, FFR_20min, FFR_25min]
    return speed_matrix, A, FFR






