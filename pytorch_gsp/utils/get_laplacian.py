import torch
from torch_scatter import scatter_add
from .loop import add_self_loops
from .sparse import eye
from .num_nodes import maybe_num_nodes
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import torch_sparse as ts
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_laplacian(A, normalization=None, dtype=None
                 ):

    assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

    
    index, edge_weight=A._indices(), A._values()
    row,col = index
    num_nodes=A.size()[0]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    edge_index=torch.cat([row.view(1,row.shape[0]),col.view(1,col.shape[0])],dim=0)
    if normalization is None:
        # L = D - A.
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = torch.cat([-edge_weight, deg], dim=0)
    elif normalization == 'sym':
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # L = I - A_norm.
        edge_index, edge_weight = add_self_loops(edge_index, -edge_weight,
                                                 fill_value=1,
                                                 num_nodes=num_nodes)
    else:
        # Compute A_norm = -D^{-1} A.
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0
        edge_weight = deg_inv[row] * edge_weight

        # L = I - A_norm.
        edge_index, edge_weight = add_self_loops(edge_index, -edge_weight,
                                                 fill_value=1,
                                                 num_nodes=num_nodes)

    row,col=edge_index
    return torch.sparse.FloatTensor(edge_index,
                  edge_weight,
                 (num_nodes,num_nodes)).coalesce()


def get_laplacian_dense(A, normalization=None, dtype=None,
                  num_nodes=None):
    

    assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

    assert(A.shape[0==A.shape[1]])
    N=A.shape[0]
    D=A.sum(axis=0)
  
    if normalization is None:
        # L = D - A.
        L=np.diag(D)-A
   
    elif normalization == 'sym':
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt=D**(-1/2)
        
        A_norm=deg_inv_sqrt*A*deg_inv_sqrt
        L=np.identity(N)-A_norm

        
    else:
        raise NotImplementedError
        # Compute A_norm = -D^{-1} A.
        

    return L

def normalize(A, normalization='sym', dtype=None, num_nodes=None):
    

    assert normalization in [ 'sym', 'rw'], 'Invalid normalization'

    edge_index, edge_weight=A._indices(), A._values()

    num_nodes=A.size()[0]

    

    row, col = edge_index[0],edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)


    if normalization == 'sym':
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # L = I - A_norm.
       
        # Compute A_norm = -D^{-1} A.
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0
        edge_weight = deg_inv[row] * edge_weight

        

    return  torch.sparse.FloatTensor(edge_index,
                  edge_weight,
                 (num_nodes,num_nodes)).coalesce()

def normalize_dense(A, normalization='sym', dtype=None,
                  num_nodes=None):
    
    assert normalization in [ 'sym', 'rw'], 'Invalid normalization'

    assert(A.shape[0==A.shape[1]])
    N=A.shape[0]
    D=A.diagonal()

    if normalization == 'sym':
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt=np.diag(D**(-1/2))
        A_norm=deg_inv_sqrt@A@deg_inv_sqrt
        L=A_norm

        
    else:
        raise NotImplementedError
        # Compute A_norm = -D^{-1} A.
        

    return L


def rescale(L, lmax=None, lmin=0):
    """Rescale Laplacian eigenvalues to [-1,1]"""
    assert(L.layout == torch.sparse_coo)
    M, M = L.size()
    L=L.float()
    if lmax is None:
        L_sp=ts.to_scipy(L._indices(), L._values(), M,M).astype(np.float32)
        #L_sp=
        lmax=abs(eigsh(L_sp, k=1, which='LM', return_eigenvectors=False)[0])
    dtype=L.dtype
    I = eye(M,dtype=dtype).to(dtype).to(device)
   
    return (L-lmin*I)/(lmax-lmin) * 2 -I 



def lmax_L(L):
    """Compute largest Laplacian eigenvalue"""
    M, M = L.size()
    L=L.float()
    L_sp=ts.to_scipy(L._indices(), L._values(), M,M).astype(np.float32)

    return abs(eigsh(L_sp, k=1, which='LM', return_eigenvectors=False)[0])

def add_self_loops(A):
    n = A.shape[0]
    return np.eye(n) + A
