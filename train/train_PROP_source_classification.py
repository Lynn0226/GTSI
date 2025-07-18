"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import accuracy_SBM as accuracy, binary_f1_score, binary_precision_score, binary_recall_score, binary_auc_score


def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    epoch_train_f1 = 0
    epoch_train_pre = 0
    epoch_train_rec = 0
    epoch_train_auc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['snapshots'].to(device)  # num x feat
        batch_e = batch_graphs.ndata['snapshots'].to(device)  # modify edata
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0;
            sign_flip[sign_flip < 0.5] = -1.0
            # randomly flip the sign of the eigenvectors during training, following Dwivedi et al. (2020)
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None

        try:
            batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
        except:
            batch_wl_pos_enc = None

        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)

        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        epoch_train_f1 += binary_f1_score(batch_scores, batch_labels)
        epoch_train_pre += binary_precision_score(batch_scores, batch_labels)
        epoch_train_rec += binary_recall_score(batch_scores, batch_labels)
        epoch_train_auc += binary_auc_score(batch_scores, batch_labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    epoch_train_f1 /= (iter + 1)
    epoch_train_pre /= (iter + 1)
    epoch_train_rec /= (iter + 1)
    epoch_train_auc /= (iter + 1)

    return epoch_loss, epoch_train_acc, epoch_train_f1, epoch_train_pre, epoch_train_rec, epoch_train_auc, optimizer


def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    epoch_test_f1 = 0
    epoch_test_pre = 0
    epoch_test_rec = 0
    epoch_test_auc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['snapshots'].to(device)
            batch_e = batch_graphs.ndata['snapshots'].to(device)  # modify edata
            batch_labels = batch_labels.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None

            try:
                batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
            except:
                batch_wl_pos_enc = None

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            epoch_test_f1 += binary_f1_score(batch_scores, batch_labels)
            epoch_test_pre += binary_precision_score(batch_scores, batch_labels)
            epoch_test_rec += binary_recall_score(batch_scores, batch_labels)
            epoch_test_auc += binary_auc_score(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        epoch_test_f1 /= (iter + 1)
        epoch_test_pre /= (iter + 1)
        epoch_test_rec /= (iter + 1)
        epoch_test_auc /= (iter + 1)

    return epoch_test_loss, epoch_test_acc, epoch_test_f1, epoch_test_pre, epoch_test_rec, epoch_test_auc


