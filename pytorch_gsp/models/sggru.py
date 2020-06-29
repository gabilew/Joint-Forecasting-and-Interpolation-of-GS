import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
import time
from pytorch_gsp.utils.gsp import (spectral_components,  Reconstruction)

class SpectralGraphForecast(nn.Module):
    """
    SpectralGraphForecast 

    Args:
        V (numpy array): eingenvectors matrix graph signal processing model (i.e.: Laplacian matrix of the graph)
        sample (numpy array): indices of in sample nodes
        freqs (numpy array): frequency components to be used in interpolation
        rnn (str, optional): predictive model: lstm, gru, 1dconv. Defaults to 'gru'. 
    """
    def __init__(self, V, sample,freqs, rnn = 'gru'):
        super(SpectralGraphForecast, self).__init__()  
     
        self.N = V.shape[0] # number of nodes in the entire graph
        self.d = len(freqs) # number of frequencies
        self.n = len(sample) # number of samples
        self.sample = sample
        if rnn == 'gru':     
            self.srnn = nn.GRU(self.d,self.d,1, batch_first=True)
            self.rnn =nn.GRU(self.n,self.n,1, batch_first=True)
        elif rnn == 'lstm':
            self.srnn = nn.LSTM(self.d,self.d,1, batch_first=True)
            self.rnn =nn.LSTM(self.n,self.n,1, batch_first=True)
        elif rnn == '1dconv':
            self.srnn = nn.Conv1d(self.d,self.d,1, batch_first=True)
            self.rnn =nn.Conv1d(self.n,self.n,1, batch_first=True)

        if self.n != self.N:
            self.interpolate = Reconstruction(V,sample,freqs, domain='freq')
            self.interpolate2 = Reconstruction(V,sample,freqs, domain='vertex')
        
        self.linear = nn.Linear(self.N*2,self.N)
      
    def forward(self, input):
        x = input[:,:,:self.n]
        x_hat = input[:,:,self.n:]
        bz, seq_len, _ = x.size()
  
        x_hat = self.srnn(x_hat)[0][:,-1,:]
        
        if self.n != self.N:
            xtilde = self.interpolate(x_hat).unsqueeze(1)
        else:
            xtilde = x_hat.unsqueeze(1)
        
        x = self.rnn(x)[0][:,-1,:]

        if self.n != self.N:
            x1 = self.interpolate2(x) 
            x1[:,self.sample] = x

        else:
            x1 = x
        x1 = x1.unsqueeze(1) 
        x1 = torch.cat((xtilde,x1),dim = 1).reshape((bz, self.N*2))
        return self.linear(x1)

class SpectralGraphForecast2(nn.Module):
    """
    SpectralGraphForecast2: combination of predictive models in both spectral and vertex domains

    Args:
        V (numpy array): eingenvectors matrix graph signal processing model (i.e.: Laplacian matrix of the graph)
        sample (numpy array): indices of in sample nodes
        freqs (numpy array): frequency components to be used in interpolation
        rnn (str, optional): predictive model: lstm, gru, . Defaults to 'gru'. 
    """
    def __init__(self, V, sample,freqs, rnn = 'gru'):

        super(SpectralGraphForecast2, self).__init__()
        
     
        self.N = V.shape[0]
        self.d = len(freqs)
        self.n = len(sample)
        self.sample = sample
        if rnn == 'gru':
     
            self.srnn = nn.GRU(self.d,self.d,1, batch_first=True)
            self.rnn =nn.GRU(self.n,self.n,1, batch_first=True)
        elif rnn == 'lstm':
            self.srnn = nn.LSTM(self.d,self.d,1, batch_first=True)
            self.rnn =nn.LSTM(self.n,self.n,1, batch_first=True)

        if self.n != self.N:
            self.interpolate = Reconstruction(V,sample,freqs, domain='freq')
            self.interpolate2 = Reconstruction(V,sample,freqs, domain='vertex')
        
   
        self.w = Parameter(torch.Tensor(self.N), requires_grad=True)
        self.w.data.fill_(0.01)

    def forward(self, input):
        x = input[:,:,:self.n]
        x_hat = input[:,:,self.n:]
        bz, seq_len, _ = x.size()
  
        x_hat = self.srnn(x_hat)[0][:,-1,:]
        
        if self.n != self.N:
            xtilde = self.interpolate(x_hat)
        else:
            xtilde = x_hat
        
        x = self.rnn(x)[0][:,-1,:]

        if self.n != self.N:
            x1 = self.interpolate2(x)             

        return torch.tanh(self.w)*xtilde + (1-torch.tanh(self.w))*x1

class model(nn.Module):
    def __init__(self, V, sample,freqs, layer, supervised = True, l1=0,l2=0, schedule_step=10):
        """
        model: model class to use the SpectralGraphForecast layer 

        Args:
        V (numpy array): eingenvector matrix graph from signal processing model (i.e.: Laplacian matrix of the graph)
        sample (numpy array): indices of in sample nodes
        freqs (numpy array): frequency components to be used in interpolation
        layer (nn.Module): SpectralGraphForecast layer
        """
        super(model, self).__init__()
        
        self.N = V.shape[0]
        self.d = len(freqs)
        self.n = len(sample)
        self.supervised = supervised
        self.sample = sample
        self.layer = layer
        self.l1 = l1
        self.l2 = l2
        self.schedule_step = schedule_step
        if not supervised:
            self.interpolate = Reconstruction(V,sample,freqs, domain='vertex')
        
    def forward(self, input):
        
        return self.layer(input)
    
    def loss(self,out,y):
        assert (self.l1+self.l2 <=1)
        assert(self.l1>=0)
        assert(self.l2>=0)
        regularization_loss = 0
        if self.l1 != 0:
                regularization_loss += self.l1*torch.nn.L1Loss()(y[:,self.sample],out[:,self.sample])
        if self.l2 != 0:
                regularization_loss += self.l2*torch.norm(y[:,self.sample]-out[:,self.sample])
        
        if not self.supervised:
            ys =  y
            y = self.interpolate(ys)    
            y[:,self.sample] = ys
        return  torch.nn.MSELoss()(y,out) + regularization_loss
  

    def schedule(self,opt):
        for param_group in opt.param_groups:
            learning_rate = param_group['lr']
        if learning_rate > 1e-5:
            lamb = lambda epoch:  0.5 if epoch%10 == 0 else 1
        else: lamb = lambda epoch:  1 if epoch%10 == 0 else 1
        
        return  torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=[lamb])


class model2(nn.Module):
    def __init__(self, V, sample,freqs, layer,l1=0,l2=0,schedule_step=10, supervised = True, unsqueeze=False):
        super(model2, self).__init__()
        """
        model2: interepolates the signal before running the layer.

        Args:
        V (numpy array): eingenvector matrix graph from signal processing model (i.e.: Laplacian matrix of the graph)
        sample (numpy array): indices of in sample nodes
        freqs (numpy array): frequency components to be used in interpolation
        layer (nn.Module): layer
        """
        self.N = V.shape[0]
        self.d = len(freqs)
        self.n = len(sample)
        self.supervised = supervised
        self.sample = sample
        self.unsqueeze = unsqueeze
        self.layer = layer 
        self.l1 = l1
        self.l2 = l2
        self.schedule_step = schedule_step
        self.interpolate2 = Reconstruction(V,sample,freqs, domain='vertex')
        if not supervised:
            self.interpolate = Reconstruction(V,sample,freqs, domain='vertex')
        self.linear = torch.nn.Linear(self.N,self.N)
        
    def forward(self, input):
        bz, seq_len, N = input.size()
        if self.unsqueeze:
            x = input.unsqueeze(dim=1)
        x = self.layer(input)
        if N < self.N:
            x1 = self.interpolate2(x)
            x1[:,self.sample] = x
        else: x1 = x
        return x1
    
    def loss(self,out,y):
        assert (self.l1+self.l2 <1)
        assert(self.l1>=0)
        assert(self.l2>=0)
        regularization_loss = 0
        if self.l1 != 0:
                regularization_loss += self.l1*torch.nn.L1Loss()(y[:,self.sample],out[:,self.sample])
        if self.l2 != 0:
                regularization_loss += self.l2*torch.norm(y[:,self.sample]-out[:,self.sample])
        
        if not self.supervised:
            ys =  y
            y = self.interpolate(ys)    
            y[:,self.sample] = ys
        return  torch.nn.MSELoss()(y,out) + regularization_loss
    

    def schedule(self,opt):
    
        for param_group in opt.param_groups:
            learning_rate = param_group['lr']
        if learning_rate > 1e-5:
            lamb = lambda epoch:  1/2 if epoch%self.schedule_step == 0 else 1
        else: lamb = lambda epoch:  1 if epoch%5 == 0 else 1
        
        return  torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=[lamb])
