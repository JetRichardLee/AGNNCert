# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 19:39:37 2024

@author: 31271
"""

import torch
import os

def evaluate(out, labels):
    """
    Calculates the accuracy between the prediction and the ground truth.
    :param out: predicted outputs of the explainer
    :param labels: ground truth of the data
    :returns: int accuracy
    """
    
    preds = out.argmax(dim=1)
    #print(preds.shape)
    #print(labels.shape)
    correct = preds == labels
    if correct.size(0)==0:
        return 0.0
    acc = int(correct.sum()) / int(correct.size(0))
    return acc



def store_checkpoint(paper, dataset, model, train_acc, val_acc, test_acc, epoch=-1):
    """
    Store the model weights at a predifined location.
    :param paper: str, the paper 
    :param dataset: str, the dataset
    :param model: the model who's parameters we whish to save
    :param train_acc: training accuracy obtained by the model
    :param val_acc: validation accuracy obtained by the model
    :param test_acc: test accuracy obtained by the model
    :param epoch: the current epoch of the training process
    :retunrs: None
    """
    save_dir = f"./checkpoints/{paper}/{dataset}"
    checkpoint = {'model_state_dict': model.state_dict(),
                  'train_acc': train_acc,
                  'val_acc': val_acc,
                  'test_acc': test_acc}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if epoch == -1:
        torch.save(checkpoint, os.path.join(save_dir, f"best_model"))
    else:
        torch.save(checkpoint, os.path.join(save_dir, f"model_{epoch}"))

def load_best_model(best_epoch, paper, dataset, model, eval_enabled):
    """
    Load the model parameters from a checkpoint into a model
    :param best_epoch: the epoch which obtained the best result. use -1 to chose the "best model"
    :param paper: str, the paper 
    :param dataset: str, the dataset
    :param model: the model who's parameters overide
    :param eval_enabled: wheater to activate evaluation mode on the model or not
    :return: model with pramaters taken from the checkpoint
    """
    print(best_epoch)
    if best_epoch == -1:
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/best_model")
    else:
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/model_{best_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])

    if eval_enabled: model.eval()

    return model

def train_model(model, edge_index, x, labels, train_mask, val_mask, test_mask, train_args):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_args["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(0, train_args["epochs"]):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=train_args["clip_max"])
        optimizer.step()
        
        with torch.no_grad():
            out = model(x, edge_index)

            # Evaluate train
        train_acc = evaluate(out[train_mask], labels[train_mask])
        test_acc = evaluate(out[test_mask], labels[test_mask])
        val_acc = evaluate(out[val_mask], labels[val_mask])

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")
        if val_acc > best_val_acc: # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(train_args["paper"], train_args["dataset"], model, train_acc, val_acc, test_acc)

        if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
            break