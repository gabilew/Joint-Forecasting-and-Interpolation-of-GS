### dataloaders
### modified from https://github.com/zhiyongc/Graph_Convolutional_LSTM/blob/master/Code_V2/HGC_LSTM%20%26%20Experiments.ipynb
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as utils
from torch_gsp.utils.gsp import compute_sample, complement
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def DataloaderMissingVal(matrix,samp, V , freqs , batch_size = 40, seq_len = 10, pred_len = 1, train_proportion = 0.7,
 valid_proportion = 0.2,T = False,seed = 10 , nan = 0,A = None):
    
    time_len = matrix.shape[0]
    max_val = matrix.max().max()
    matrix =  matrix / max_val

    if samp is None:
        samp = np.arange(matrix.shape[1])
    recurrent_features, labels,labels_unknown = [], [], []
  
    if T :
        assert(freqs is not None)
        transformed = []

    
    np.random.seed(seed)
    
    if nan != 0:
        n = int(nan*matrix.size)
        nodes = np.random.choice(range(matrix.shape[1]),n,replace = True)
        time =  np.random.choice(range(matrix.shape[0]),n,replace = True)
        features_nan=np.copy(matrix.values)
        features_nan[time,nodes] = 0
    
        for node,t in zip(nodes,time):      
            neighbors = np.nonzero(A[node])[0]   
            features_nan[np.array([t]), np.array([node])] = np.mean(features_nan[t, neighbors])
        
    
    for i in range(time_len - seq_len - pred_len):  
        features = features_nan[i:i+seq_len]        
        recurrent_features.append(features)

        if T :         
           
            Vf = V[:, freqs]
            Psi = np.zeros(Vf.shape[0])
            Psi[sample] = 1
            Psi = np.diag(Psi)
            Psi = Psi[:, sample]
            Tx = (Vf.T@Ps@(features.T)).T
            transformed.append(Tx)        
      
        
        labels.append(matrix.iloc[i+seq_len:i+seq_len+pred_len].values)
        labels_unknown.append(features_nan[i+seq_len:i+seq_len+pred_len])
             
    recurrent_features, labels, labels_unknown = np.asarray(recurrent_features), np.asarray(labels),np.asarray(labels_unknown)
    
    
    if T:
        transformed = np.asarray(transformed)
        
    if len(labels.shape) ==3:
        labels = labels[:,-1,:]
        labels_unknown = labels_unknown[:,-1,:]

    samp_size = recurrent_features.shape[0]    
    index = np.arange(samp_size, dtype = int)    

    train_index = int(np.floor(samp_size * train_proportion))
    valid_index = int(np.floor(samp_size * ( train_proportion + valid_proportion)))          
   

    train_data, train_label = recurrent_features[:train_index], labels[:train_index]
    valid_data, valid_label = recurrent_features[train_index:valid_index], labels[train_index:valid_index]    
    test_data, test_label = recurrent_features[valid_index:], labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data) , torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data) , torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data) , torch.Tensor(test_label)

    if T:
        train_transf = torch.Tensor(transformed[:train_index]) 
        valid_transf = torch.Tensor(transformed[train_index:valid_index]) 
        test_transf = torch.Tensor(transformed[valid_index:]) 
        train_data = torch.cat((train_data, train_transf),dim = 2)
        valid_data = torch.cat((valid_data, valid_transf),dim = 2)
        test_data = torch.cat((test_data, test_transf),dim = 2)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, drop_last = True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size = batch_size, shuffle=True, drop_last = True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = len(test_label), shuffle=False, drop_last = False)

    return train_dataloader, valid_dataloader, test_dataloader, max_val

def NoisyDataloader(matrix,samp, V , freqs , batch_size = 40, seq_len = 10, pred_len = 1, train_proportion = 0.7,
     valid_proportion = 0.2,T = False,sigma = 0,seed = 10 ):

    time_len = matrix.shape[0]
    max_val = matrix.max().max()
    matrix =  matrix / max_val

    if samp is None:
        samp = np. arange(matrix.shape[1])
    
    recurrent_features, labels,labels_unknown = [], [], []
  
    if T :
        assert(freqs is not None)
        transformed = []
    
    if sigma != 0:
        sigma *= np.std(matrix.values)
        noise = np.random.randn(*matrix.shape)*sigma    

    for i in range(time_len - seq_len - pred_len):  
        features = matrix.iloc[i:i+seq_len].values+ noise[i:i+seq_len]        
        recurrent_features.append(features)

        if T :    
            Vf = V[:, freqs]            
            Psi = np.zeros(Vf.shape[0])
            Psi[samp] = 1
            Psi = np.diag(Psi)
            Psi = Psi[:, samp]
            Tx = (Vft@Psi@(features.T)).T
            transformed.append(Tx)             
            
        
        labels.append(matrix.iloc[i+seq_len:i+seq_len+pred_len].values)             
    recurrent_features, labels, labels_unknown = np.asarray(recurrent_features), np.asarray(labels),np.asarray(labels_unknown)
    
    
    if T:
        transformed = np.asarray(transformed)        
    
    samp_size = recurrent_features.shape[0]    
    index = np.arange(samp_size, dtype = int)    

    train_index = int(np.floor(samp_size * train_proportion))
    valid_index = int(np.floor(samp_size * ( train_proportion + valid_proportion)))
    
    if len(labels.shape) ==3:
        labels = labels[:,-1,:]
   
    train_data, train_label = recurrent_features[:train_index], labels[:train_index]
    valid_data, valid_label = recurrent_features[train_index:valid_index], labels[train_index:valid_index]    
    test_data, test_label = recurrent_features[valid_index:], labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label) 
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label) 

    if T:
        train_transf = torch.Tensor(transformed[:train_index])
        valid_transf = torch.Tensor(transformed[train_index:valid_index])
        test_transf = torch.Tensor(transformed[valid_index:])
        train_data = torch.cat((train_data, train_transf),dim = 2)
        valid_data = torch.cat((valid_data, valid_transf),dim = 2)
        test_data = torch.cat((test_data, test_transf),dim = 2)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, drop_last = True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size = batch_size, shuffle=True, drop_last = True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = len(test_label), shuffle=False, drop_last = False)

    return train_dataloader, valid_dataloader, test_dataloader, max_val




def SampledDataloader(matrix,samp, V , freqs , batch_size = 40, seq_len = 10, pred_len = 1, train_proportion = 0.7,
 valid_proportion = 0.2,T = False, sampling='', A = None, unsup = False):
    """
    SampledDataloader: perform the sampling procedure on the graph signals and create the dataloader object

    Args:
        matrix (pandas DataFrame): [description]
        samp (np array): [description]
        V (2D np array): [description]
        freqs (np array): [description]
        batch_size (int, optional): batch size. Defaults to 40.
        seq_len (int, optional): size of historical data. Defaults to 10.
        pred_len (int, optional): number of future samples. Defaults to 1.
        train_proportion (float, optional): percentage of data used in training. Defaults to 0.7.
        valid_proportion (float, optional): percentage of data used in validation. Defaults to 0.2.
        T (bool, optional): if Fourier transform should be applied. Defaults to False.
        sampling (str, optional): method of sampling "reduce" returns data with only in-sample data wheres "knn" returns data interpolated
         based on neighbors in adjacency. Defaults to ''.
        A ([type], optional): Adjacency matrix. Must be passed only if sampling method is "knn" . Defaults to None.
        unsup (bool, optional): If the dataset will be used in a semi-supervised learning. Defaults to False.

    Returns:
        [type]: [description]
    """

    assert(sampling in ['', 'reduce', 'knn']
  
    time_len = matrix.shape[0]

    max_val = matrix.max().max()
    matrix =  matrix / max_val
    
    recurrent_features, labels,labels_unknown = [], [], []
  
    if T :
        assert(freqs is not None)
        transformed = []

    if samp is None:
        samp = np.arange(0,matrix.shape[1],1)


    time_knn = time.time()
    
    if sampling == 'knn':
        unknown = complement(samp,matrix.shape[1])
        features_unknown = np.copy(matrix.values)
        features_unknown[:,unknown] = np.mean(matrix.values[:100,samp])
        for node in unknown:
            neighbors = np.nonzero(A[node])[0]
            for t in range(features_unknown.shape[0]):
            
                features_unknown[np.array([t]), np.array([node])] = np.mean(features_unknown[t, neighbors])
    time_knn = time.time()-time_knn
    time_T = 0
    for i in range(time_len - seq_len - pred_len):
        
        if sampling == 'reduce':
            features = matrix.iloc[i:i+seq_len].values[:,samp]
        elif sampling == 'knn':
            features = features_unknown[i:i+seq_len]
                
        else:
            features = matrix.iloc[i:i+seq_len].values 
                
 
        
        recurrent_features.append(features)
        time_pre = time.time()
        if T :
            assert(sampling =='reduce')
            Vf = V[:, freqs]            
            Psi = np.zeros(Vf.shape[0])
            Psi[samp] = 1
            Psi = np.diag(Psi)
            Psi = Psi[:, samp]
            Tx = (Vft@Psi@(features.T)).T
            transformed.append(Tx)
        time_T += time.time()-time_pre
        if unsup:
            labels_unknown.append(features_unknown[i+seq_len+pred_len-1:i+seq_len+pred_len])             
        
        labels.append(matrix.iloc[i+seq_len+pred_len-1:i+seq_len+pred_len].values)              
    recurrent_features, labels, labels_unknown = np.asarray(recurrent_features), np.asarray(labels),np.asarray(labels_unknown)
    
    
    if T:
        transformed = np.asarray(transformed)
        
    
    samp_size = recurrent_features.shape[0]    
    index = np.arange(samp_size, dtype = int)   

    train_index = int(np.floor(samp_size * train_proportion))
    valid_index = int(np.floor(samp_size * ( train_proportion + valid_proportion)))
    
    if unsup:
        train_data, train_label = recurrent_features[:train_index], labels_unknown[:train_index]
        valid_data, valid_label = recurrent_features[train_index:valid_index], labels_unknown[train_index:valid_index]  
       
    else:

        train_data, train_label = recurrent_features[:train_index], labels[:train_index]
        valid_data, valid_label = recurrent_features[train_index:valid_index], labels[train_index:valid_index]    
    test_data, test_label = recurrent_features[valid_index:], labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data) , torch.Tensor(train_label) 
    valid_data, valid_label = torch.Tensor(valid_data) , torch.Tensor(valid_label) 
    test_data, test_label = torch.Tensor(test_data) , torch.Tensor(test_label) 

    if T:
        train_transf = torch.Tensor(transformed[:train_index]) 
        valid_transf = torch.Tensor(transformed[train_index:valid_index]) 
        test_transf = torch.Tensor(transformed[valid_index:]) 
        train_data = torch.cat((train_data, train_transf),dim = 2)
        valid_data = torch.cat((valid_data, valid_transf),dim = 2)
        test_data = torch.cat((test_data, test_transf),dim = 2)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data,  valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, drop_last = True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size = batch_size, shuffle=True, drop_last = True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = len(test_label), shuffle=False, drop_last = False)

    if T is not None:
        return train_dataloader, valid_dataloader, test_dataloader, max_val
    elif sampling == 'knn':
        return train_dataloader, valid_dataloader, test_dataloader, max_val
    else:
        return train_dataloader, valid_dataloader, test_dataloader, max_val


def PrepareSequence(data, seq_len = 10, pred_len = 1):
  
    time_len = data.shape[0]    
    sequences, labels = [], []  
    for i in range(time_len - seq_len - pred_len):       
        sequences.append(data[i:i+seq_len])        
        labels.append(data[i+seq_len+pred_len-1:i+seq_len+pred_len])             
    return  np.asarray(sequences), np.asarray(labels)
    
    

def SplitData(data, label = None, seq_len = 10, pred_len = 1, train_proportion = 0.7,
 valid_proportion = 0.2, shuffle = False):
            
    
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
        return train_data ,valid_data, test_data





def Dataloader(data, label, gpu = True,  batch_size = 40, suffle = False):
    
    if gpu:
        data, label = torch.Tensor(data).cuda(), torch.Tensor(label ).cuda()
    else:
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

    x = matrix.T[sample]
    Vf = V[:, freqs]
    Psi = np.zeros(Vf.shape[0])
    Psi[sample] = 1
    Psi = np.diag(Psi)
    Psi = Psi[:, sample]
    Tx = (Vf.T@Psi@x).T
    return Tx
       
class DataPipeline:
    def __init__(self,  sample, V , freqs ,seq_len, pred_len):

        self.sample = sample
        self.V = V
        self.freqs = freqs
        self.seq_len = seq_len
        self.pred_len = pred_len

    def fit(self,train_data,sample_label = False):
        
        if  not sample_label:
            self.max, train_data = MaxScaler(train_data)
        else:
            self.max, _ = MaxScaler(train_data[:,self.sample])
            train_data/=self.max
     
        
        train_data_freqs = Preprocessing_GFT(train_data,self.sample, self.V , self.freqs )
        train_X, train_y = PrepareSequence(train_data, seq_len = self.seq_len, pred_len = self.pred_len)
        train_X_freqs, _ = PrepareSequence(train_data_freqs, seq_len = self.seq_len, pred_len = self.pred_len)
        if sample_label:
            train_y = train_y.T[self.sample]
            train_y = train_y.T
      
        return np.concatenate((train_X[:,:,self.sample], train_X_freqs), axis=-1), train_y

    def transform(self, data, sample_label = False):
        data/=self.max
        data_freqs = Preprocessing_GFT(data,self.sample, self.V , self.freqs)
        X, y = PrepareSequence(data, seq_len = self.seq_len, pred_len = self.pred_len)
        X_freqs, _ = PrepareSequence(data_freqs, seq_len = self.seq_len, pred_len = self.pred_len)
        if sample_label:
            y = y.T[self.sample]
            y = y.T
        return np.concatenate((X[:,:,self.sample], X_freqs), axis=-1), y

    


    
        


