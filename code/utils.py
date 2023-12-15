import torch
import numpy as np
import scipy.sparse as sp
import scipy
from torch_sparse import SparseTensor


def accuracy(output, label):
    """ Return accuracy of output compared to label.
    Parameters
    ----------
    output:
        output from model (torch.Tensor)
    label:
        node label (torch.Tensor)
    """
    #print(output.max(1)[1])
    preds = output.max(1)[1].type_as(label)
    correct = preds.eq(label).double()
    correct = correct.sum()
    return correct / len(label)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_mx_to_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    rows = torch.from_numpy(sparse_mx.row).long()
    cols = torch.from_numpy(sparse_mx.col).long()
    values = torch.from_numpy(sparse_mx.data)
    return SparseTensor(row=rows, col=cols, value=values, sparse_sizes=torch.tensor(sparse_mx.shape))

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features#.todense()

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
    adj = normalize_adj(adj).todense()
    adj = torch.from_numpy(adj)
    return features, adj, labels

def normalize_adj(adj):    # 这部分就是计算D-1/2AD-1/2
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # 计算每一个节点的度
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.   # d_inv_sqrt是1的话则对应位置置为0，否则不变
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

# Count the number of samples in each class
def count_each_class_num(labels):
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict

def generate_permutation(labels, args):
    '''
    Generate permutation for training, validating and testing data.
    '''
    N = labels.shape[0]
    each_class_num = count_each_class_num(labels)
    training_each_class_num = {}  ## number of labeled samples for each class

    for label in each_class_num.keys():
        training_each_class_num[label] = args.ratio
        valid_num = 500
        test_num = 1000
    if args.dataset in ['Cornell', 'Tesax', 'Wisconsin']:
        valid_num = np.floor(N * 0.2)
        test_num = np.floor(N * 0.2)

    # index of labeled and unlabeled samples
    train_mask = torch.from_numpy(np.full((N), False))
    valid_mask = torch.from_numpy(np.full((N), False))
    test_mask = torch.from_numpy(np.full((N), False))

    # shuffle the data
    data_idx = np.random.permutation(range(N))

    # Get training data
    for idx in data_idx:
        label = labels[idx]
        if (training_each_class_num[label] > 0):
            training_each_class_num[label] -= 1
            train_mask[idx] = True
    for idx in data_idx:
        if train_mask[idx] == True:
            continue
        if (test_num > 0):
            test_num -= 1
            test_mask[idx] = True
        elif (valid_num > 0):
            valid_num -= 1
            valid_mask[idx] = True
    return train_mask, valid_mask, test_mask


def load_data(data_path, ptb_path, dataset, args):
    print("Loading {} dataset...".format(dataset))
    ruin_datapath = './ruin_dataset/' + dataset
    # if args.ptb_feat == False:
    #     feature = sp.load_npz(data_path+dataset+"/feat.npz")
    # else:
    #     feature = sp.load_npz(ptb_path+"feat/"+dataset+"/"+"feat_"+str(args.ratio)+".npz")
    # if dataset == "breast_cancer" or dataset == "digits" or dataset == "wine" or dataset == "wikics":
    #     feature = feature.todense()
    # else:
    #     feature = preprocess_features(feature)

    # label = torch.LongTensor(np.load(data_path+dataset+"/label.npy"))
    # label = torch.LongTensor(label)

    # idx_train = np.load(data_path + dataset + "/train.npy")
    # idx_val = np.load(data_path + dataset + "/val.npy")
    # idx_test = np.load(data_path + dataset + "/test.npy")

    feature, adj, label = loadMatData(ruin_datapath)
    feature = preprocess_features(feature)
    feature = torch.FloatTensor(np.array(feature))
    label = label.squeeze()
    idx_train, idx_val, idx_test = generate_permutation(label, args)
    label = torch.LongTensor(label)

    ori_view1 = sp.load_npz(data_path+dataset+"/"+args.name_view1+".npz")       
    ori_view2 = sp.load_npz(data_path+dataset+"/"+args.name_view2+".npz")
    ori_view1_indice = torch.load(data_path+dataset+"/"+args.indice_view1+".pt")
    ori_view2_indice = torch.load(data_path+dataset+"/"+args.indice_view2+".pt")

    print(ptb_path+"add/"+dataset+"/"+str(args.ratio)+"_1.npz", "ruin")
    
    if args.add:
        if args.flag == 1:
            ori_view1 = sp.load_npz(ptb_path+"add/"+dataset+"/"+str(args.flag_ratio)+"_1.npz")
        elif args.flag == 2:
            ori_view2 = sp.load_npz(ptb_path+"add/"+dataset+"/"+str(args.flag_ratio)+"_2.npz")
        elif args.flag == 3:
            ori_view1 = sp.load_npz(ptb_path+"add/"+dataset+"/"+str(args.flag_ratio)+"_1.npz")
            ori_view2 = sp.load_npz(ptb_path+"add/"+dataset+"/"+str(args.flag_ratio)+"_2.npz")
    elif args.dele:
        if args.flag == 1:
            ori_view1 = sp.load_npz(ptb_path+"dele/"+dataset+"/"+str(args.flag_ratio)+"_1.npz")
        elif args.flag == 2:
            ori_view2 = sp.load_npz(ptb_path+"dele/"+dataset+"/"+str(args.flag_ratio)+"_2.npz")
        elif args.flag == 3:
            ori_view1 = sp.load_npz(ptb_path+"dele/"+dataset+"/"+str(args.flag_ratio)+"_1.npz")
            ori_view2 = sp.load_npz(ptb_path+"dele/"+dataset+"/"+str(args.flag_ratio)+"_2.npz")
            
    ori_view1 = sparse_mx_to_torch_sparse_tensor(normalize_adj(ori_view1)) 
    ori_view2 = sparse_mx_to_torch_sparse_tensor(normalize_adj(ori_view2))
    # print(ori_view1.shape,"ruin")
    return DataSet(dataset=args.dataset, x=feature, y=label, view1=ori_view1, view2=ori_view2,
                   view1_indice=ori_view1_indice, view2_indice=ori_view2_indice,
                   idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

class DataSet():
    def __init__(self, dataset, x, y, view1, view2, view1_indice, view2_indice, idx_train, idx_val, idx_test):
        self.dataset = dataset
        self.x = x
        self.y = y
        self.view1 = view1
        self.view2 = view2
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.num_node = x.size(0)
        self.num_feature = x.size(1)
        self.num_class = int(torch.max(y)) + 1
        self.v1_indices = view1_indice
        self.v2_indices = view2_indice
        print(self.num_class, self.num_node, self.num_feature)

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.view1 = self.view1.to(device)
        self.view2 = self.view2.to(device)
        self.v1_indices = self.v1_indices.to(device)
        self.v2_indices = self.v2_indices.to(device)
        return self

    def normalize(self, adj):
        if self.dataset in ["wikics", "ms"]:
            adj_ = (adj + adj.t())
            normalized_adj = adj_
        else:
            adj_ = (adj + adj.t())
            normalized_adj = self._normalize(adj_ + torch.eye(adj_.shape[0]).to(adj.device).to_sparse())
        return normalized_adj

    def _normalize(self, mx):
        mx = mx.to_dense()
        rowsum = mx.sum(1) + 1e-6  # avoid NaN
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx.to_sparse()
