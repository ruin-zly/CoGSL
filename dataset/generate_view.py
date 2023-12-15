import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity as cos
import scipy.sparse as sp
import scipy
from scipy.linalg import fractional_matrix_power, inv
import sys

argv = sys.argv
k=20
alpha=0.1

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def loadMatData(data_path):
    data = scipy.io.loadmat(data_path)
    features = data['X']  # .dtype = 'float32'
    # features = torch.from_numpy(features).float()
    labels = data['Y']
    adj = data['adj']
    return features, adj, labels

def knn(feat, num_node, k, data_name, view_name):
    adj = np.zeros((num_node, num_node), dtype=np.int64)
    dist = cos(feat)
    col = np.argpartition(dist, -(k + 1), axis=1)[:, -(k + 1):].flatten()
    adj[np.arange(num_node).repeat(k + 1), col] = 1
    adj = sp.coo_matrix(adj)
    sp.save_npz("./"+data_name+"/"+view_name+"_knn.npz", adj)
    # sp.save_npz("../ptb/add/" + data_name + "/20_1.npz", adj)


def Adj(adj, data_name, view_name):
    adj = sp.coo_matrix(adj)
    sp.save_npz("./"+data_name+"/"+view_name+"_adj.npz", adj)
    # sp.save_npz("../ptb/add/" + data_name + "/"+str(ratio)+"_1.npz", adj)

def diff(adj, alpha, data_name, view_name):   
    d = np.diag(np.sum(adj, 1))                                    
    dinv = fractional_matrix_power(d, -0.5)                       
    at = np.matmul(np.matmul(dinv, adj), dinv)                      
    adj = alpha * inv((np.eye(adj.shape[0]) - (1 - alpha) * at))   
    adj = sp.coo_matrix(adj)
    sp.save_npz("./"+data_name+"/"+view_name+"_diff.npz", adj)
    # sp.save_npz("../ptb/add/" + data_name + "/"+str(ratio)+"_2.npz", adj)

data_name = "UAI"
view_name = "v2"  # v1 or v2
view_type = "knn"  # knn adj diff


ruin_datapath = '/home/zhongly/single_view_dataset/' + data_name
feat, adj, _ = loadMatData(ruin_datapath)
a = adj

# ratio = 0.1
# add_path = data_name+"_add_"+str(ratio)+".npz"
# adj = sp.load_npz("/home/zhongly/DIB-RGCN/CoGSL-main/ptb/"+add_path)
# adj = normalize_adj(adj)
# a = adj.A

# adj = sp.load("./"+data_name+"/ori_adj.npz")
# feat = sp.load("./"+data_name+"/feat.npz")
# a = adj.A
num_node = adj.shape[0]
if a[0, 0] == 0:
    a = a + np.eye(num_node)
    print("self-loop!")
adj = a
if view_type == "knn":  # set k
    knn(feat, num_node, k, data_name, view_name)
elif view_type == "adj":
    Adj(adj, data_name, view_name)
elif view_type == "diff":  # set alpha: 0~1
    diff(adj, alpha, data_name, view_name)
