
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as utils
from torch_gsp.utils.gsp import compute_sample, complement


def PrepareSequence(data, seq_len = 10, pred_len = 1):
  
    time_len = data.shape[0]    
    sequences, labels = [], []  
    for i in range(time_len - seq_len - pred_len):       
        sequences.append(data[i:i+seq_len])        
        labels.append(data[i+seq_len+pred_len-1:i+seq_len+pred_len])             
    return  np.asarray(sequences), np.asarray(labels)
    
    

def SplitData(data, label = None, seq_len = 10, pred_len = 1, train_proportion = 0.7,
 valid_proportion = 0.2, shuffle = False):

    max_value = np.max(data)       
    data /= max_value
    samp_size = data.shape[0]     
    if label is not None:
        assert(label.shape[0] == samp_size)

    index = np.arange(samp_size, dtype = int)   
    train_index = int(np.floor(samp_size * train_proportion))
    valid_index = int(np.floor(samp_size * ( train_proportion + valid_proportion)))    
    
    if label is not None:
        train_data, train_label = data[:train_index+pred_len-1], label[:train_index+pred_len-1]
        valid_data, valid_label = data[train_index-seq_len:valid_index+pred_len-1], label[train_index-seq_len:valid_index+pred_len-1]    
        test_data, test_label = data[valid_index-seq_len:], label[valid_index-seq_len:]
        return (train_data, train_label), (valid_data, valid_label), (test_data, test_label)

    else:
        train_data = data[:train_index+pred_len-1]
        valid_data = data[train_index-seq_len:valid_index+pred_len-1]    
        test_data = data[valid_index-seq_len:]       
        return train_data ,valid_data, test_data, max_value



def Dataloader(data, label,  batch_size = 40, suffle = False):
   
    data, label = torch.Tensor(data), torch.Tensor(label )
    dataset = utils.TensorDataset(data, label)    
    dataloader = utils.DataLoader(dataset, batch_size = batch_size, shuffle=suffle, drop_last = True)
    return dataloader


def Preprocessing_hop_interp(matrix, A ,sample):  
    
    unknown = complement(sample,matrix.shape[1])
    features_unknown = np.copy(matrix.values)
    features_unknown[:,unknown] = np.mean(matrix.values[:100,sample])
    for node in unknown:
        neighbors = np.nonzero(A[node])[0]
        for t in range(features_unknown.shape[0]):            
            features_unknown[np.array([t]), np.array([node])] = np.mean(features_unknown[t, neighbors])    
    return features_unknown


def MaxScaler(data):
    max_value = np.max(data)
    return max_value, data/max_value

def Preprocessing_GFT(matrix,sample, V , freqs ):

    x = matrix.T
    Vf = V[:, freqs]
    Psi = np.zeros((V.shape[0],x.shape[1]))
    Psi[sample] = x
    Tx = (Vf.T@Psi).T
    return Tx
       
class DataPipeline:
    def __init__(self,  sample, V , freqs ,seq_len, pred_len, gft = True):
        """
        DataPipeline: perform the sampling procedure on the graph signals and create the dataloader object
        Args:        
        sample (np array): list of graph indices 
        V (2D np array): Laplacian eigenvector matrix
        freqs (np array): list of frequency indices        
        seq_len (int, optional): size of historical data. Defaults to 10.
        pred_len (int, optional): number of future samples. Defaults to 1.        
        gft (bool, optional): if Fourier transform should be applied. Defaults to False.     
        """

        self.sample = sample
        self.V = V
        self.freqs = freqs
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.gft = gft

    def fit(self,train_data,sample_label = True, batch_size=40, shuffle=True):
    
        train_X, train_y = PrepareSequence(train_data, seq_len = self.seq_len, pred_len = self.pred_len)

        if self.gft:
            train_data_freqs = Preprocessing_GFT(train_data[:,self.sample],self.sample, self.V , self.freqs )        
            train_X_freqs, _ = PrepareSequence(train_data_freqs, seq_len = self.seq_len, pred_len = self.pred_len)
            train_X = np.concatenate((train_X[:,:,self.sample], train_X_freqs), axis=-1)

        if sample_label:
            train_y = train_y.T[self.sample]
            train_y = train_y.T
      
        return Dataloader(train_X, train_y, batch_size, shuffle)

    def transform(self, data, sample_label = True, batch_size=40,shuffle=True):
        
        X, y = PrepareSequence(data, seq_len = self.seq_len, pred_len = self.pred_len)

        if self.gft:
            data_freqs = Preprocessing_GFT(data[:,self.sample],self.sample, self.V , self.freqs)        
            X_freqs, _ = PrepareSequence(data_freqs, seq_len = self.seq_len, pred_len = self.pred_len)
            
            X = np.concatenate((X[:,:,self.sample], X_freqs), axis=-1)
        if sample_label:
            y = y.T[self.sample]
            y = y.T
        return Dataloader(X, y, batch_size, shuffle)

    


    
        


