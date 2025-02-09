# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import sys
sys.path.append("models/")
#from mlp import MLP
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import copy
from utils import evaluate, store_checkpoint, load_best_model, train_model
from sklearn.model_selection import train_test_split

device =  "cuda" if torch.cuda.is_available() else "cpu"
class HashAgent():
    def __init__(self,h="md5",T=30):
        '''
            h: the hash function in "md5","sha1","sha256"
            T: the subset amount
        '''

        super(HashAgent, self).__init__()
        self.T = T
        self.h= h 
        
            
    def hash_node(self,u):
        #"""
        hexstring = hex(u)
        hexstring= hexstring.encode()
        if self.h == "md5":
            hash_device = hashlib.md5()
        elif self.h == "sha1":
            hash_device = hashlib.sha1()
        elif self.h == "sha256":
            hash_device = hashlib.sha256()
        hash_device.update(hexstring)
        I = int(hash_device.hexdigest(),16)%self.T
        
        return I
    
    def generate_node_subgraphs(self, edge_index, x, y):
        
        subgraphs = []
        
        original = edge_index
        nodes = range(x.shape[0])

        V= x.shape[0]
                    
        for i in range(self.T):
            subgraphs.append(Data(
                        x = x,
                        y = y,
                        edge_index = []
                    ))
        for i in range(len(original[0])):
            u=original[0,i]
            v=original[1,i]
            I = self.hash_node(u)
            subgraphs[I].edge_index.append([u,v])
            
        new_subgraphs = []
        for i in range(self.T):
            if len(subgraphs[i].edge_index)==0:
                continue
            subgraphs[i].edge_index = torch.tensor(subgraphs[i].edge_index,dtype=torch.int64).transpose(1,0)
            new_subgraphs.append(subgraphs[i])
            
        return new_subgraphs

    def generate_graph_subgraphs(self, edge_index, x, y):
        
        subgraphs = []
        
        original = edge_index
        nodes = range(x.shape[0])
        zerox = torch.zeros(x[0].size()).reshape(1,-1)
        V= x.shape[0]
        xs = [[] for i in range(self.T)]
        mappings = []
        for i in range(self.T):
            subgraphs.append(Data(
                        x = zerox,
                        y = y,
                        edge_index = []
                    ))
        #print(V)
        for i in range(x.shape[0]):
            I = self.hash_node(i)
            mappings.append(subgraphs[I].x.shape[0])
            subgraphs[I].x = torch.cat((subgraphs[I].x,x[i].reshape(1,-1)),dim=0)
            subgraphs[I].edge_index.append([mappings[i],0])
            
        for i in range(len(original[0])):
            
            u=original[0,i]
            v=original[1,i]
            I = self.hash_node(u)
            if self.hash_node(v)==I:
                subgraphs[I].edge_index.append([mappings[u],mappings[v]])
            
        new_subgraphs = []
        for i in range(self.T):
            if subgraphs[i].x.shape[0]<=1:
                continue
            subgraphs[i].edge_index = torch.tensor(subgraphs[i].edge_index,dtype=torch.int64).transpose(1,0)

            new_subgraphs.append(subgraphs[i])
            
        return new_subgraphs
    
    def generate_amazon_subgraphs(self, edge_index, x, y):
        
        subgraphs = []
        
        original = edge_index
        V= x.shape[0]
        adding_table = [None for _ in range(self.T)]
        #Following the idea of cluster-GCN, we keep only a subpart of edges in the subgraph
        for i in range(self.T):
            adding_table[i] = [0]+random.sample(range(1,self.T),5)
            for j in range(len(adding_table[i])):
                adding_table[i][j] = (adding_table[i][j]+i)%self.T
            subgraphs.append(Data(
                        x = x,
                        y = None,
                        edge_index = []
                    ))
            
            
        print("Dividing edges")
        for i in range(len(original[0])):
            if i %1000000==0:
                print(i//1000000,"M / ",len(original[0])//1000000,"M")
            u=original[0,i]
            v=original[1,i]
            I = self.hash_node(u)
            I2 = self.hash_node(v)
            for j in range(len(adding_table[I2])):
                if I2 == adding_table[I][j]:
                    subgraphs[I].edge_index.append([u,v])
                    break
                
        new_subgraphs = []
        for i in range(self.T):
            if len(subgraphs[i].edge_index)==0:
                continue
            print("subgraph:",i,"/",self.T)
            subgraphs[i].edge_index = torch.tensor(subgraphs[i].edge_index,dtype=torch.int64).transpose(1,0)
            new_subgraphs.append(subgraphs[i])

        return new_subgraphs
    
class RobustNodeClassifier():
    def __init__(self,model,Hasher,edge_index, x, y, train_mask, val_mask, test_mask,num_labels):
        super(RobustNodeClassifier, self).__init__()
        
        self.model = model
        self.Hasher = Hasher
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.edge_index = edge_index
        self.x = x
        self.y = y
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_labels= num_labels
    def load_model(self,path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def train(self, train_args ):
        subgraphs = self.Hasher.generate_node_subgraphs(self.edge_index, self.x, self.y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_args["lr"])
        criterion = torch.nn.CrossEntropyLoss()
    
        best_val_acc = 0.0
        best_epoch = 0
    
        for epoch in range(0, train_args["epochs"]):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(self.x, self.edge_index)
            loss = criterion(out[self.train_mask], self.y[self.train_mask])
            for i in range(len(subgraphs)):
                out_sub = self.model(subgraphs[i].x,subgraphs[i].edge_index)
                loss+=criterion(out_sub[self.train_mask], self.y[self.train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=train_args["clip_max"])
            optimizer.step()
            
            train_acc = 0
            test_acc = 0
            val_acc = 0
            with torch.no_grad():
                out_train,_ = self.vote(self.train_mask)
                out_val,_ = self.vote(self.val_mask)
                out_test,_ = self.vote(self.test_mask)
                train_acc = evaluate(out_train, self.y[self.train_mask])
                val_acc = evaluate(out_val, self.y[self.val_mask])
                test_acc = evaluate(out_test, self.y[self.test_mask])
                
            print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")
            if val_acc > best_val_acc: # New best results
                print("Val improved")
                best_val_acc = val_acc
                best_epoch = epoch
                store_checkpoint("robust_n/"+train_args["paper"], train_args["dataset"]+"/{}".format(self.Hasher.T), self.model, train_acc, val_acc, test_acc)

            if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
                break
        
     
    def test(self ):   
        
        out_test,M = self.vote(self.test_mask)
        test_acc = evaluate(out_test, self.y[self.test_mask])
        return test_acc, M
        
    def vote(self, mask):
        subgraphs = self.Hasher.generate_node_subgraphs(self.edge_index, self.x, self.y)
        V_test = self.x[mask].shape[0]
        votes = torch.zeros((V_test,self.num_labels))
        self.model.eval()
        for i in range(len(subgraphs)):
            out_sub = self.model(subgraphs[i].x,subgraphs[i].edge_index)
            preds = out_sub[mask].argmax(dim=1)
            for j in range(V_test):
                votes[j,preds[j]]+=1
 
        vote_label = votes.argmax(dim=1)
        M =torch.zeros(V_test)
        for i in range(V_test):
            votes[i,vote_label[i]]=-votes[i,vote_label[i]]
        second_label = votes.argmax(dim=1)
        for i in range(V_test):
            votes[i,vote_label[i]]=-votes[i,vote_label[i]]
        for i in range(V_test):
            if vote_label[i]>second_label[i]:
                M[i] = (votes[i,vote_label[i]]-votes[i,second_label[i]]-1)//2
            else:
                M[i] = (votes[i,vote_label[i]]-votes[i,second_label[i]])//2
        return votes, M
    
   
    
class RobustGraphClassifier():
    def __init__(self,model,Hasher,graphs,labels,train_mask, val_mask, test_mask,num_labels):
        super(RobustGraphClassifier, self).__init__()
        self.model = model
        self.Hasher = Hasher
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_labels= num_labels
        self.graphs=graphs
        self.labels = torch.tensor(labels)
        self.subgraphs=[]
        for i in range(len(graphs)):
            self.subgraphs.append(self.Hasher.generate_graph_subgraphs(graphs[i].edge_index,graphs[i].x,graphs[i].y))
            
    def load_model(self,path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def enlarge_dataset(self, graphs):
        new_graphs = []
        ys = []
        for i in range(len(graphs)):
            subgraphs = self.Hasher.generate_graph_subgraphs(graphs[i].edge_index,graphs[i].x,graphs[i].y)
            for j in range(len(subgraphs)):
                new_graphs.append(subgraphs[j].to(self.device))
                ys.append(subgraphs[j].y.to(self.device))
            new_graphs.append(graphs[i].to(self.device))
            ys.append(graphs[i].y.to(self.device))
        return new_graphs, ys
    
    
    def train(self, train_args ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_args["lr"])
        criterion = torch.nn.CrossEntropyLoss()
    
        best_val_acc = 0.0
        best_train_acc = 0.0
        best_epoch = 0
        
        train_graphs = self.graphs[self.train_mask]
        entrain_graphs,ys = self.enlarge_dataset(train_graphs)
        for epoch in range(0, train_args["epochs"]):
            self.model.train()
            optimizer.zero_grad()
            loss = torch.zeros(1).to(self.device)
            for i in range(len(entrain_graphs)):
                out = self.model(entrain_graphs[i].x,entrain_graphs[i].edge_index)
                loss+=criterion(out, ys[i].to(torch.long))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=train_args["clip_max"])
            optimizer.step()
            
            train_acc = 0
            test_acc = 0
            val_acc = 0
            with torch.no_grad():
                out_train,_ = self.vote(self.train_mask)
                out_val,_ = self.vote(self.val_mask)
                #out_test,_ = self.vote(self.test_mask)
                train_acc = evaluate(out_train, self.labels[self.train_mask])
                val_acc = evaluate(out_val, self.labels[self.val_mask])
                #test_acc = evaluate(out_test, self.labels[self.test_mask])


            print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss.item():.4f}")
            if val_acc == best_val_acc and train_acc>best_train_acc: # New best results
                print("Train improved")
                #best_val_acc = val_acc
                best_train_acc = train_acc
                best_epoch = epoch
                store_checkpoint("robust_n/"+train_args["paper"], train_args["dataset"]+"/{}".format(self.Hasher.T), self.model, train_acc, val_acc, test_acc)

            if val_acc > best_val_acc: # New best results
                print("Val improved")
                best_val_acc = val_acc
                best_train_acc = train_acc
                best_epoch = epoch
                store_checkpoint("robust_n/"+train_args["paper"], train_args["dataset"]+"/{}".format(self.Hasher.T), self.model, train_acc, val_acc, test_acc)

            if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
                break
        
     
    def test(self ):   
        
        out_test,M = self.vote(self.test_mask)
        test_acc = evaluate(out_test, self.labels[self.test_mask])
        return test_acc, M
        
    def vote(self, mask):
        #test_graphs = self.graphs[mask]

        G_test = len(self.graphs[mask])
        idxs = np.array([i for i in range(len(self.graphs))])
        test_id = idxs[mask]
        
        votes = torch.zeros((G_test,self.num_labels))
        M =torch.zeros(G_test)
        
        
        self.model.eval()
        for i in range(G_test):
            subgraphs = self.subgraphs[test_id[i]]
            for j in range(len(subgraphs)):
                out_sub = self.model(subgraphs[j].x.to(self.device),subgraphs[j].edge_index.to(self.device)).cpu()
                preds = out_sub[0].argmax(dim=0)
                votes[i,preds]+=1
        
        vote_label = votes.argmax(dim=1)
        for i in range(G_test):
            votes[i,vote_label[i]]=-votes[i,vote_label[i]]
        second_label = votes.argmax(dim=1)
        for i in range(G_test):
            votes[i,vote_label[i]]=-votes[i,vote_label[i]]
        for i in range(G_test):
            if vote_label[i]>second_label[i]:
                M[i] = (votes[i,vote_label[i]]-votes[i,second_label[i]]-1)//2
            else:
                M[i] = (votes[i,vote_label[i]]-votes[i,second_label[i]])//2
        return votes, M
      
class RobustAmazonNodeClassifier():
    def __init__(self,model,Hasher,edge_index, x, y, train_idx, valid_idx, test_idx,num_labels):


        super(RobustAmazonNodeClassifier, self).__init__()
        self.model = model.to(device)
        self.Hasher = Hasher
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.edge_index = edge_index
        self.x = x.to(device)
        self.y = y
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.num_labels= num_labels
        self.subgraphs = Hasher.generate_amazon_subgraphs(self.edge_index,self.x,self.y)
        self.T = self.Hasher.T
    def load_model(self,path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def train(self, train_args ):
        subgraphs = self.subgraphs#Hasher.generate_node_subgraphs(self.edge_index, self.x, self.y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_args["lr"])
        criterion = torch.nn.CrossEntropyLoss()
    
        best_val_acc = 0.0
        best_epoch = 0
        for epoch in range(0, train_args["epochs"]):
            self.model.train()
            optimizer.zero_grad()
            loss = 0.0
            
            train_batch= random.sample(range(0,self.T),5)
            for i in range(len(train_batch)):
                batch_graph = copy.deepcopy(subgraphs[train_batch[i]]).to(device)
                print("training: ", i," / ", len(train_batch))
                out_sub = self.model(self.x,batch_graph.edge_index)
                loss+=criterion(out_sub[self.train_idx], self.y[self.train_idx].to(device))
                del batch_graph
                torch.cuda.empty_cache()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=train_args["clip_max"])
            optimizer.step()
            
            train_acc = 0
            test_acc = 0
            val_acc = 0
            print(f"Epoch: {epoch},  train_loss: {loss:.4f}")
            if epoch%25==15:
                with torch.no_grad():
                    outs = self.vote_multi([self.train_idx,self.valid_idx,self.test_idx])
                    out_train = outs[0]
                    out_val = outs[1]
                    out_test = outs[2]
                    train_acc = evaluate(out_train, self.y[self.train_idx])
                    val_acc = evaluate(out_val, self.y[self.valid_idx])
                    test_acc = evaluate(out_test, self.y[self.test_idx])
                
                print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")
                if val_acc > best_val_acc: # New best results
                    print("Val improved")
                    best_val_acc = val_acc
                    best_epoch = epoch
                    store_checkpoint("robust_n/"+train_args["paper"], train_args["dataset"]+"/{}".format(self.Hasher.T), self.model, train_acc, val_acc, test_acc)

                if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
                    break
        
     
    def test(self ):   
        
        out_test,M = self.vote(self.test_mask)
        test_acc = evaluate(out_test, self.y[self.test_mask])
        return test_acc, M
        
    def vote(self, mask):
        subgraphs = self.subgraphs
        V_test = self.x[mask].shape[0]
        votes = torch.zeros((V_test,self.num_labels))
        self.model.eval()
        for i in range(len(subgraphs)):
            test_subgraph = copy.deepcopy(subgraphs[i]).to(device)
            out_sub = self.model(self.x,test_subgraph.edge_index)
            preds = out_sub[mask].argmax(dim=1)
            for j in range(V_test):
                votes[j,preds[j]]+=1
            del test_subgraph
            torch.cuda.empty_cache()
 
        vote_label = votes.argmax(dim=1)
        M =torch.zeros(V_test)
        for i in range(V_test):
            votes[i,vote_label[i]]=-votes[i,vote_label[i]]
        second_label = votes.argmax(dim=1)
        for i in range(V_test):
            votes[i,vote_label[i]]=-votes[i,vote_label[i]]
        for i in range(V_test):
            if vote_label[i]>second_label[i]:
                M[i] = (votes[i,vote_label[i]]-votes[i,second_label[i]]-1)//2
            else:
                M[i] = (votes[i,vote_label[i]]-votes[i,second_label[i]])//2
        return votes, M
    
    def vote_multi(self, masks):
        subgraphs = self.subgraphs
        val_batch= random.sample(range(0,self.T),10)
        V_test = self.x.shape[0]
        votes = torch.zeros((V_test,self.num_labels))
        self.model.eval()
        for i in range(len(val_batch)):
            print("val: ", i, " / ",len(val_batch))
            test_subgraph = copy.deepcopy(subgraphs[val_batch[i]]).to(device)
            out_sub = self.model(self.x,test_subgraph.edge_index)
            preds = out_sub.argmax(dim=1)
            for j in range(V_test):
                votes[j,preds[j]]+=1
            del test_subgraph
            torch.cuda.empty_cache()
            
        return [votes[mask] for mask in masks]
