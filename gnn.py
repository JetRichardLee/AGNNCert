# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 19:39:09 2024

@author: 31271
"""

import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool,SAGEConv, GATConv,GINConv 

from torch import Tensor

from torch_geometric.utils import add_remaining_self_loops,to_dense_adj
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

#a =GINConv()
class NodeGCN6(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes,hidden_size=20):
        super(NodeGCN6, self).__init__()
        self.embedding_size=hidden_size*6
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.conv4 = GCNConv(hidden_size, hidden_size)
        self.relu4 = ReLU()
        self.conv5 = GCNConv(hidden_size, hidden_size)
        self.relu5 = ReLU()
        self.conv6 = GCNConv(hidden_size, hidden_size)
        self.relu6 = ReLU()
        
        self.lin = Linear(hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []

        
        out1 = self.conv1(x, edge_index,edge_weight = None)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)+out1
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)+out2
        stack.append(out3)
        
        out4 = self.conv4(out3, edge_index)
        out4 = torch.nn.functional.normalize(out4, p=2, dim=1)  # this is not used in PGExplainer
        out4 = self.relu4(out4)+out3
        stack.append(out4)
        
        out5 = self.conv5(out4, edge_index)
        out5 = torch.nn.functional.normalize(out5, p=2, dim=1)  # this is not used in PGExplainer
        out5 = self.relu5(out5)+out4
        stack.append(out5)
        
        out6 = self.conv6(out5, edge_index)
        out6 = torch.nn.functional.normalize(out6, p=2, dim=1)  # this is not used in PGExplainer
        out6 = self.relu6(out6)+out5
        stack.append(out6)

        input_lin = torch.cat(stack, dim=1)

        return out6
class NodeGCN(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes,hidden_size=20):
        super(NodeGCN, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(3*hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []

        out1 = self.conv1(x, edge_index,edge_weight = None)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin

class GraphGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes,hidden_size=32):
        super(GraphGCN, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_size*3, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final
    
    def embedding(self, x, edge_index):
            
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)
        
        stack = torch.cat(stack, dim=1)
        input_lin = global_mean_pool(stack,batch=None)

        return input_lin
 
class NodeGSAGE(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes,hidden_size=20):
        super(NodeGSAGE, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = SAGEConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = SAGEConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(3*hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index.to(torch.int64))
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin

class GraphGSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes,hidden_size=32):
        super(GraphGSAGE, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = SAGEConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = SAGEConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_size*3, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final
    
    def embedding(self, x, edge_index):
            
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)
        
        stack = torch.cat(stack, dim=1)
        input_lin = global_mean_pool(stack,batch=None)

        return input_lin
 
class NodeGAT(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes,hidden_size=20):
        super(NodeGAT, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = GATConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GATConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GATConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(3*hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin

class GraphGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes,hidden_size=32):
        super(GraphGAT, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = GATConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GATConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GATConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_size*3, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final
    
    def embedding(self, x, edge_index):
            
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)
        
        stack = torch.cat(stack, dim=1)
        input_lin = global_mean_pool(stack,batch=None)

        return input_lin
 
    
 

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout=dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        edge_index, _ = add_remaining_self_loops(edge_index)
        x = self.conv1(features, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout,training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def count_arr(predictions, nclass):
    nodes_n=predictions.shape[0]
    counts = np.zeros((nodes_n,nclass), dtype=int)
    for n,idx in enumerate(predictions):
        counts[n,idx] += 1
    return counts

class SmoothGCN(GCN):
    '''Multi-smoothing Smoothing model for GCN. Also suitable for other GNNs.'''
    def __init__(self,in_channels, out_channels, hidden_channels, dropout,config,device):
        super(SmoothGCN, self).__init__(in_channels, out_channels, hidden_channels, dropout)
        self.config = config
        self.device=device
        self.nclass=out_channels
        self.p_e, self.p_n = torch.tensor(config['p_e']), torch.tensor(config['p_n'])

    def perturbation(self, adj_dense):
        '''Using upper triangle adjacency matrix to Perturb the edge first, and then the nodes.'''
        size = adj_dense.shape
        assert (torch.triu(adj_dense)!=torch.tril(adj_dense).t()).sum()==0
        adj_triu = torch.triu(adj_dense,diagonal=1)
        adj_triu = (adj_triu==1) * torch.bernoulli(torch.ones(size).to(self.device)*(1 - self.p_e))
        # deleted edges: (torch.triu(adj_dense)>0).sum()-(adj_triu > 0).sum()
        adj_triu = adj_triu.mul(torch.bernoulli(torch.ones(size[0]).to(self.device)*(1 - self.p_n)))
        # deleted nodes: (torch.bernoulli(torch.ones(size[0]).to(self.device)*(1 - self.p_n))==0).sum()
        adj_perted = adj_triu + adj_triu.t()
        # total deleted edges: (torch.triu(adj_dense)>0).sum()-(torch.triu(adj_perted) > 0).sum()
        return adj_perted

    def forward_perturb(self, features: Tensor, edge_index: Tensor) -> Tensor:
        """ Estimate the model with smoothing perturbed samples """
        # features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph adjacency matrix of shape [2, num_edges]
        with torch.no_grad():
            adj_dense = torch.squeeze(to_dense_adj(edge_index))
            adj_dense = self.perturbation(adj_dense)
            edge_index = torch.nonzero(adj_dense).t()
            # deleted_nodes=[]
            # for i in range(adj_dense.shape[0]):
            #     if (adj_dense[i,:]==0).all():
            #         deleted_nodes.append(i)
            # len(deleted_nodes)
        return self.forward(features, edge_index)

    def smoothed_precit(self, features, edge_index, num):
        """ Sample the base classifier's prediction under smoothing perturbation of the input x.
        num: number of samples to collect (N)
        return: top2: the top 2 classes, and the per-class counts
        """
        counts = np.zeros((features.shape[0], self.nclass), dtype=int)
        for i in tqdm(range(num), desc='Processing MonteCarlo'):
            predictions = self.forward_perturb(features, edge_index).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2]
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        return top2, count1, count2