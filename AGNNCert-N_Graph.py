# -*- coding: utf-8 -*-

from gnn import GraphGCN,GraphGAT,GraphGSAGE
from datasets.dataset_loader import load_graph_data

from node_hash import HashAgent, RobustGraphClassifier
import torch 
import numpy as np
import os 
import random

import time 

paper = "GCN"
#paper = "GAT"
#paper = "GSAGE"

dataset = "PROTEINS"
#dataset = "DD"
#dataset = "AIDS"
#dataset = "Mutagenicity"

if dataset == "Mutagenicity":
    num_train = 1000
    num_val=400
else:
    num_train = 250
    num_val=100
    
train_args = {
        "dataset": dataset,
        "paper": "GCN",
        "lr" : 0.001,
        "epochs" : 200,
        "clip_max" : 2.0,
        "batch_size": 64,
        "early_stopping": 100,
        "seed" : 42,
        "eval_enabled" : True
    }
print(train_args["dataset"])
print(train_args["seed"])

graphs,num_x,num_labels,mask_spilt,labels = load_graph_data(dataset)
device =  "cuda" if torch.cuda.is_available() else "cpu"

from utils import evaluate, store_checkpoint, load_best_model, train_model
train_mask = mask_spilt[0]
val_mask = mask_spilt[1]
test_mask = mask_spilt[2]

if paper == "GCN":
    model = GraphGCN(num_x, num_labels)
elif paper == "GAT":
    model = GraphGAT(num_x, num_labels)
elif paper == "GSAGE":
    model = GraphGSAGE(num_x, num_labels)

    
t1=time.time()
T=50
hasher = HashAgent(h="md5",T=T)
r_model =  RobustGraphClassifier(model.to(device),hasher,graphs,labels,train_mask, val_mask, test_mask,num_labels)
path = "./checkpoints/robust_n/GCN/{}/{}/best_model".format(dataset,T)


#if False:
if os.path.exists(path):
    r_model.load_model(path)
else:
    r_model.train(train_args)
     
labels= torch.tensor(labels)
test_labels = labels[test_mask]
out_test,M = r_model.vote(test_mask)

test_acc = evaluate(out_test, test_labels)
print(test_acc)

test_preds = out_test.argmax(dim=1)
print(sum(M)/len(M)) 
print(M)
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

