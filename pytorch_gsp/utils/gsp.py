import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import scipy

from sklearn.metrics.pairwise import rbf_kernel



def complement(S,N):
    V = set(np.arange(0,N,1))
    return np.array(list(V-set(S)))


class Reconstruction(nn.Module):
    def __init__(self,V, sample, freqs, domain='vertex',use_original_set = False, device = 'cuda'):
        """
        GSP reconstruction of Graph signals

        Args:
            V (numpy array): eigenvector matrix of Laplacian or adjacency. This matrix is expected to be orthonormal.
            sample (list-like): list of indices of in-sample nodes
            freqs (list): number of list of indices of 
            domain (str, optional): [description]. domain of the graph signal. Options are vertex or spectral'. Defaults to 'vertex'.
            use_original_set (bool, optional): [description]. Defaults to False.
        """

        super(Reconstruction,  self).__init__()
        assert(domain in ['vertex','spectral'])
        if domain == 'vertex':

            interp = Interpolator(V, sample, freqs)
        elif domain == 'spectral':
            interp= Interpolator(V, sample, freqs, freq=True)
        
          
        self.Interp = torch.Tensor(interp).to(device)
        self.N = V.shape[0]

        if use_original_set:
            self.sample = sample
        else:
            self.sample = None

    def forward(self,x):
        x0 = x
        n_dim = len(x.size())
        if n_dim == 3:
            bz, seq_len, n = x.size()
            x = x.T
        
            x = x.reshape((n, bz*seq_len))        
            x = torch.matmul(self.Interp,x)
            x = x.reshape((self.N,seq_len,bz)).T 

        else:
            bz,  n = x.size()
            x = x.T
        
            x = x.reshape((n, bz))        
            x = torch.matmul(self.Interp,x)
            x = x.reshape((self.N,bz)).T 
         
        return x
 


def corrMatrix(A, x): 
    """
    corrMatrix compute an adjacency matrix with radial basis function entries 

    Args:
        A (2D numpy array): adjacency matrix
        x (2D numpy array): signals to be used to compute correlations

    Returns:
        2D numpy array: adjacency matrix
    """
    cor = rbf_kernel(x.T/10)  
       
    A = cor*(A)       
      
    e, _ = np.linalg.eigh(A)
    A/=np.max(e)
    return  A-np.diag(A.diagonal())



def spectral_components(A, x, return_vectors = True,lap = True, norm = False):
    """
    spectral_components:  compute the index of spectral components with largest magnitude in a set of graph signals

    Args:
        A (2d numpy array): adjacency matrix
        x (2d numpy array): graph signals with time in the rows and nodes in the columns
        return_vectors (bool, optional): [description]. Defaults to True.
        lap (bool, optional): If it is the spectral components are computed using the laplacian. Defaults to True.
        norm (bool, optional): [description]. If the matrix should be normalized as $D^{-1/2}AD^{-1/2}$.

    Returns:
        [type]: [description]
    """
        
    if lap:
        if norm:
            d = 1/np.sqrt(A.sum(axis=1))
            D=np.diag(d)
            I = np.diag(np.ones(A.shape[0]))
            L = I - D@A@D
        else:
            D = np.diag(A.sum(axis=1))
            L = D - A    
    else:
        if norm:
            d = 1/np.sqrt(A.sum(axis=1))
            D=np.diag(d)
            I = np.diag(np.ones(A.shape[0]))
            L = D@A@D
        else: L = A
    lambdas, V = np.linalg.eigh(L)  
    
    
    energy = np.abs(V.T@x.T).T
    index = []
    for y in energy:
    
        index.append(list(np.argsort(y)))
    
    ocorrencias = {i:0 for i in range(x.shape[1]) }
    for y in index:
        for i in y:
            ocorrencias[i]+= y.index(i)
    
    F_global= np.argsort([ocorrencias[oc] for oc in ocorrencias])[::-1]
    if return_vectors:
        return F_global, V
    else:
        return F_global


def Interpolator(V, sample,  freqs, freq = False):	
 
    
    Psi = np.zeros(Vf.shape[0])
    Psi[sample] = 1 #transpose  of the sampling operator \Psi
    Psi = np.diag(Psi)   
    Psi = Psi[:, sample]
    I = np.diag(np.ones(Vf.shape[0]))
    inv = scipy.linalg.inv(Vf.T@Psi@Vf)
    if freq == False:
        pseudoi = inv@Vf.T@Psi
    else:
        pseudoi = inv
        
    interp = np.dot(Vf, pseudoi)
    Psi_bar = I - Psi
    s = np.linalg.svd(np.dot(Psi_bar, Vf), compute_uv=False)
    if np.max(s)>1:
        print("Samling is not admissable")
        return None
      
    return interp



class KNN(nn.Module):
    def __init__(self,A,sample, matrix):
        super(KNN,self).__init__()
        N = A.shape[0]
        self.unknown = complement(sample,N)
        self.mask = np.mean(matrix.values[:,sample])
    def forward(self, input):
        if len(input.size()) == 2:
            input[:,self.unknown] = self.mask
        elif len(input.size()) == 3:
            input[:,:,self.unknown] = self.mask
        elif len(input.size()) == 4:
            input[:,:,:,self.unknown] = self.mask
        x = input
        for node in self.unknown:
            neighbors = np.nonzero(A[node])[0]
            x[:,:,[node]] = torch.mean(x[:,:, neighbors], dim=-1)
        return x




def greedy_e_opt(Uf,  S):
    """
    code from https://github.com/georgosgeorgos/GraphSignalProcessing, please refer to this repository

    MIT License

    Copyright (c) 2018 Giorgio Giannone

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    greedy_e_opt: sample S nodes from a set of size N where N is the number of rows in matrix Uf
    
    Args:
        Uf (2D numpy array): truncated eigenvector matrix with N rows. Columns correspond to the selected eigenvectors
        S (int): sample size

    Returns:
        sample: list of indices of selected nodes
    """
    index_set = set()
    sample=[]

    n = Uf.shape[0] - 1
    k = 0
    I = np.diag(np.ones(Uf.shape[0]))
    while len(index_set) < S:
        i = -1
        i_best = -1
        old_list = []
        sigma_best = np.inf
        while i < n:
            i = i + 1
            if i in index_set:
                continue
            else:
                Ds_list = np.zeros(Uf.shape[0])
                ix = sample + [i]
                Ds_list[ix] = 1

                Ds = np.diag(Ds_list)
                Ds_bar = I - Ds
                DU = np.dot(Ds_bar, Uf)
                s = np.linalg.svd(DU, compute_uv=False)
                sigma_max = max(s)

                if sigma_max < sigma_best and sigma_max != -np.inf:
                    sigma_best = sigma_max
                    i_best = i
        k = k + 1       
        index_set.add(i_best)
        sample.append(i_best)
    return sample
