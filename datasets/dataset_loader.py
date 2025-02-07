# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:59:53 2024

@author: 31271
"""

import torch
from torch_geometric.datasets import Planetoid, Amazon
import numpy as np
import scipy.sparse as sp
from numpy.random.mtrand import RandomState

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

def matri_to_index(A):
    V = A.shape[0]
    edge_index_0 = []
    edge_index_1 = []
    
    for i in range(V):
        for j in range(i,V):
            if A[i,j]==1:
                edge_index_0.append(i)
                edge_index_1.append(j)
                if i!=j:
                    edge_index_0.append(j)
                    edge_index_1.append(i)
    return np.array([edge_index_0,edge_index_1])
                
def matri_to_index_directed(A):
    V = A.shape[0]
    edge_index_0 = []
    edge_index_1 = []
    
    for i in range(V):
        for j in range(V):
            if A[i,j]==1:
                edge_index_0.append(i)
                edge_index_1.append(j)
    return np.array([edge_index_0,edge_index_1])



#To load the data, we divide the train, validate and test dataset by amount for each class
#instead of ratio, to ensure a balanced training set  

def load_node_data(name,num_train=400,num_val=100,num_test=None,directed=False):
    if name == "CiteSeer" or name == "PubMed":
        dataset = Planetoid(root='./' + name + '/', name=name,num_train_per_class=50)
        #_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _device = torch.device('cpu')
        data = dataset[0].to(_device)
        num_classes = dataset.num_classes

    elif name == "computers":
        dataset = Amazon(root='./' + name + '/', name=name)
        #_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _device = torch.device('cpu')
        data = dataset[0].to(_device)
        data.train_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.val_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.test_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        
        num_classes = dataset.num_classes
    elif name == "Cora-ML":
        data_name = './datasets/cora_ml.npz'
        with np.load(data_name, allow_pickle = True) as loader:
            loader = dict(loader)
            A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                               loader['adj_indptr']), shape=loader['adj_shape'])
            adj = A.toarray()
            X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])
            x = X.toarray()
            y = loader.get('labels')
            if not directed:
                edge_index = np.load("core_ml_edge_index.npy")
            else:
                edge_index = np.load("core_ml_edge_index_d.npy")
                
        data = Data(x=torch.tensor(x,dtype=torch.float32),edge_index=torch.tensor(edge_index),y=torch.tensor(y))
        data.train_mask = torch.zeros(y.size, dtype=torch.bool)
        data.val_mask = torch.zeros(y.size, dtype=torch.bool)
        data.test_mask = torch.zeros(y.size, dtype=torch.bool)
        
        num_classes = len(np.unique(y))
       
    
        
    prng = RandomState(12) # Make sure that the permutation is always the same, even if we set the seed different
    
    data.train_mask.fill_(False)
    data.val_mask.fill_(False)
    data.test_mask.fill_(False)
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = prng.permutation(idx)
        train_idx = idx[:num_train]
        val_idx = idx[num_train:num_train+num_val]
        if num_test is None:
            test_idx = idx[num_train+num_val:-1]
        
        else:
            test_idx = idx[num_train+num_val:num_train+num_val+num_test]
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
    if name == 'NELL':
        data.x = data.x.to_dense()
        num_node_features = data.x.shape[1]
    else:
        num_node_features = data.x.shape[1]
    return data, num_node_features, num_classes


def load_graph_data(name,num_train=100,num_val=50,num_test=50,directed=False):
    
    graphs = TUDataset(root="./datasets/", name=name,use_node_attr=True)
    num_node_features = graphs[0].x.shape[1]
    #print(graphs[0].x.shape)
    #print(graphs[0].y.shape)
    #print(graphs[0].edge_index.shape)
    ys = [graphs[i].y.item() for i in range(len(graphs))]
    num_classes = len(np.unique(ys))
    seed=12
    rng = np.random.RandomState(seed)
    
    train_mask = torch.zeros(len(graphs), dtype=torch.bool)
    val_mask = torch.zeros(len(graphs), dtype=torch.bool)
    test_mask = torch.zeros(len(graphs), dtype=torch.bool)
    
    for c in range(num_classes):
        idx = (torch.tensor(ys) == c).nonzero(as_tuple=False).view(-1)
        idx = rng.permutation(idx)
        train_idx = idx[:num_train]
        val_idx = idx[num_train:num_train+num_val]
        test_idx = idx[num_train+num_val:-1]
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
    return graphs, num_node_features, num_classes,[train_mask,val_mask,test_mask],ys

