# -*- coding: utf-8 -*-
from gnn import NodeGCN,NodeGAT,NodeGSAGE
from datasets.dataset_loader import load_node_data

from edge_hash import HashAgent, RobustNodeClassifier
import torch 
import numpy as np
import os 
import random
import time 

paper = "GCN"

#dataset = "CiteSeer"
dataset = "Cora-ML"
#dataset = "PubMed"
#dataset = "computers"
T=300

if dataset == "PubMed":
    num_train=2000
    num_val = 600
elif dataset == "computers":
    num_train=400
    num_val = 133
else:
    num_train= 150
    num_val = 50
    
train_args = {
        "dataset": dataset,
        "paper": paper,
        "lr" : 0.002,
        "epochs" : 200,
        "clip_max" : 2.0,
        "batch_size": 64,
        "early_stopping": 100,
        "seed" : 42,
        "eval_enabled" : True
    }
print(train_args["dataset"])
print(train_args["seed"])

data,num_x,num_labels = load_node_data(dataset,num_train=num_train,num_val=num_val)
from utils import evaluate, store_checkpoint, load_best_model, train_model

if paper == "GCN":
    model = NodeGCN(num_x, num_labels)
elif paper == "GAT":
    model = NodeGAT(num_x, num_labels)
elif paper == "GSAGE":
    model = NodeGSAGE(num_x, num_labels)

x = torch.tensor(data.x)
edge_index = torch.tensor(data.edge_index)
labels = torch.tensor(data.y)
train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

T=300

t1= time.time()
hasher = HashAgent(h="md5",T=T)
r_model = RobustNodeClassifier(model,hasher,edge_index, x, labels, train_mask, val_mask, test_mask,num_labels)

path = "./checkpoints/robust_e/{}/{}/{}/best_model".format(paper,dataset,T)
if os.path.exists(path):
    r_model.load_model(path)
else:
    r_model.train(train_args)
    r_model.load_model(path)
     
test_labels = labels[test_mask]
out_test,M = r_model.vote(test_mask)
test_acc = evaluate(out_test, test_labels)
print(test_acc)

test_preds = out_test.argmax(dim=1)

count = {}
for i in range(len(M)):
    c = "{}".format(int(M[i])) 
    if test_preds[i]!=test_labels[i]:
        continue
    if not c in count:
        count[c]=1 
    else:
        count[c]+=1 
sorted_count = dict(sorted(count.items(),key=lambda item:int(item[0])))
print(sorted_count)
print(len(M))
