import numpy as np
import scipy.sparse as sp
import torch


def get_khop_indices(k, view):
    view = (view.A > 0).astype("int32")
    view_ = view
    for i in range(1, k):
        view_ = (np.matmul(view_, view.T)>0).astype("int32")
    view_ = torch.tensor(view_).to_sparse()
    return view_.indices()
    
def topk(k, adj):
    pos = np.zeros(adj.shape)
    for i in range(adj.shape[0]):
      one = adj[i].nonzero()[0]
      if len(one)>k:
        oo = np.argsort(-adj[i, one])
        sele = one[oo[:k]]
        pos[i, sele] = adj[i, sele]
      else:
        pos[i, one] = adj[i, one]
    return pos

#####################
## get k-hop scope ##
## take citeseer   ##
#####################
dataset = "Computers"
adj = sp.load_npz("./"+dataset+"/v1_adj.npz")
indice = get_khop_indices(1, adj)
torch.save(indice, "./"+dataset+"/v1_1.pt")
print("yes")
#####################
## get top-k scope ##
## take citeseer   ##
#####################
adj = sp.load_npz("./"+dataset+"/v2_knn.npz").todense()
kn = topk(1, adj)
print("yes")
kn = sp.coo_matrix(kn)
indice = get_khop_indices(1, kn)
torch.save(indice, "./"+dataset+"/v2_1.pt")
print("yes")
