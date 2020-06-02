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

            matrix_rec, Ds, Ps = compute_reconstruction_matrix(V, sample, freqs)
        elif domain == 'spectral':
            matrix_rec, Ds, Ps = compute_reconstruction_matrix(V, sample, freqs, freq=True)
        
          
        self.Interp = torch.Tensor(matrix_rec).to(device)
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
           # if self.sample is not None:
            #    x[:,:,self.sample]=x0[:,:,self.sample]
        else:
            bz,  n = x.size()
            x = x.T
        
            x = x.reshape((n, bz))        
            x = torch.matmul(self.Interp,x)
            x = x.reshape((self.N,bz)).T 
         #   if self.sample is not None:
          #      x[:,self.sample]=x0[:,self.sample]
        return x
 


def PseudoI(V,sample, K):
   
    I, Ps, Ds = compute_reconstruction_matrix(V, sample, K)
    return I

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
    spectral_components:  compute the index of spectral components

    Args:
        A ([type]): [description]
        x ([type]): [description]
        return_vectors (bool, optional): [description]. Defaults to True.
        lap (bool, optional): [description]. Defaults to True.
        norm (bool, optional): [description]. Defaults to False.

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


def compute_reconstruction_matrix(U, index_list, F_list, freq = False):
	
    Uf = U[:,F_list]
    n = Uf.shape[0]
    Ds = np.zeros(n)
    Ds[index_list] = 1
    Ds = np.diag(Ds)
    Uft = Uf.conj().T
    
    Ps = Ds[:, index_list]
    I = np.diag(np.ones(n))
    matrix = p(Uft, Ds, Uf)
    matrix_inv = scipy.linalg.inv(matrix)
    if freq == False:
        Q = p(matrix_inv, Uft, Ps)
    else:
        Q = matrix_inv
        
    matrix_r = np.dot(Uf, Q)
    Ds_bar = I - Ds
    s = np.linalg.svd(np.dot(Ds_bar, Uf), compute_uv=False)
    check = max(s) < 1
    
      
    return matrix_r, Ds, Ps


def p(a, b, c):
	res = np.dot(a, np.dot(b, c))
	return res


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

class Compute_sample(nn.Module):
    def __init__(self,U,  index_list, F_list, device = 'gpu'):
        super(Compute_sample,self).__init__()

        n = U.shape[0]
        Ds = np.zeros(n)
        Ds[index_list] = 1
        self.Ds = np.diag(Ds)
        self.Ps = self.Ds[:,index_list]
        self.F_list = F_list
        Uf = U[:,self.F_list]
        self.U = torch.Tensor(U).to(device)
        self.mat = torch.Tensor(np.dot(self.Ps.T, Uf)).to(device)
        
        self.index_list = index_list

    def forward(self,y):
        s = torch.matmul(self.U.T, y.T).T
      
        s_f = s[self.F_list]
        y_sampled = torch.matmul(self.mat, s_f.T).T

        return y_sampled


def compute_sample(U,  index_list, F_list):
    n = U.shape[0]
    Ds = np.zeros(n)
    Ds[index_list] = 1
    Ds = np.diag(Ds)
    Ps = Ds[:,index_list]
    
    def f(y):
        s = np.dot(U.conj().T, y)
        Uf = U[:,F_list]
        s_f = s[F_list]
        y_sampled = p(Ps.T, Uf, s_f)
        return y_sampled
    return f


def greedy_e_opt(Uf,  S):
    """
    greedy_e_opt: sample S nodes from a set of size N where N is the number of rows in matrix Uf

    Args:
        Uf (2D numpy array): truncated eigenvector matrix with N rows. Columns correspond to the selected eigenvectors
        S (int): sample size

    Returns:
        index_list: list of indices of selected nodes
    """
    index_set = set()
    index_list=[]

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
                ix = index_list + [i]
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
        index_list.append(i_best)
    return index_list
