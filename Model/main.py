import torch
import torch.nn as nn
import numpy as np
import random
import os
import math
from sklearn import metrics
from torch_cluster import random_walk
from torch.utils.data import dataset,dataloader

test_metric = []
DP=DataProcess(kfolds, rw_param, adj_threshold, common_set, train_set, test_set)
testLoader = DP.tr_va_te_data_set(test_set['edge'], test_set['label'], batch, -1)
for i in range(kfolds):
    ES=EarlyStopping("./models/ZC_Cmodel_fold_%d.pkl"%i,patience=3)
    Cmodel=CNet(input_dim,C_model_param['hidden_size'],C_model_param['n_heads'],rw_param['rw_len']+1).to(device)
    optimizer = torch.optim.Adam(Cmodel.parameters(), lr=C_model_param['lr'], weight_decay=C_model_param['weight_decay'])
    trainLoader = DP.tr_va_te_data_set(train_set['edge_train_%d'%i], train_set['label_train_%d'%i], batch, i)
    validLoader = DP.tr_va_te_data_set(train_set['edge_valid_%d'%i], train_set['label_valid_%d'%i], batch, i)
    for e in range(C_model_param['epochs']):
        Cmodel.train()
        loss_total=0
        for _, _, p_rw_m, p_rw_d, n_rw_m, n_rw_d in trainLoader:
            p_n_output,p_r_output=Cmodel(p_rw_m.float().to(device),p_rw_d.float().to(device))
            n_n_output,n_r_output=Cmodel(n_rw_m.float().to(device),n_rw_d.float().to(device))
            loss=contrastive_loss(p_n_output,p_r_output,n_n_output,n_r_output,nce_temp,calpha)
            loss_total+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('fold:'+str(i+1)+';epoch'+str(e+1)+':'+str(loss_total))
        ES(loss_total,Cmodel)
        if ES.earlystop :
            print('contrastive learning stop')
            break
    ES=EarlyStopping("./models/ZC_Pmodel_fold_%d.pkl"%i,patience=3)
    Pmodel=PNet(input_dim,C_model_param['hidden_size'],P_model_param['hidden_channle'],P_model_param['num_group'],
                P_model_param['gate_treshold'],P_model_param['dropout_rate']).to(device)
    Cmodel.load_state_dict(torch.load("./models/ZC_Cmodel_fold_%d.pkl"%i))
    optimizer = torch.optim.Adam(Pmodel.parameters(), lr=P_model_param['lr'], weight_decay=P_model_param['weight_decay'])
    cost=nn.CrossEntropyLoss()
    for e in range(P_model_param['epochs']):
        Pmodel.train()
        for data, label, p_rw_m, p_rw_d, _, _ in trainLoader:
            n_x,r_x=Cmodel(p_rw_m.float().to(device),p_rw_d.float().to(device))
            output=Pmodel(torch.cat([p_rw_m[:,0:1,:].float().to(device),p_rw_d[:,0:1,:].float().to(device)],dim=1),
                          torch.cat([n_x.detach(),r_x.detach()],dim=1))
            loss=cost(output,label.long().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Pmodel.eval()
        with torch.no_grad():
            loss_total=0
            for data, label, p_rw_m, p_rw_d, _, _ in validLoader:
                n_x,r_x=Cmodel(p_rw_m.float().to(device),p_rw_d.float().to(device))
                output=Pmodel(torch.cat([p_rw_m[:,0:1,:].float().to(device),p_rw_d[:,0:1,:].float().to(device)],dim=1),
                              torch.cat([n_x.detach(),r_x.detach()],dim=1))
                loss=cost(output,label.long().to(device))
            loss_total+=loss.item()
        print('fold:'+str(i+1)+';epoch'+str(e+1)+':'+str(loss_total))
        ES(loss_total,Pmodel)
        if ES.earlystop :
            print('stop')
            break
    l_m,p_m=[],[]
    Pmodel.load_state_dict(torch.load("./models/ZC_Pmodel_fold_%d.pkl"%i))
    Pmodel.eval()
    with torch.no_grad():
        for data, label, p_rw_m, p_rw_d, _, _ in testLoader:
            n_x,r_x=Cmodel(p_rw_m.float().to(device),p_rw_d.float().to(device))
            output=Pmodel(torch.cat([p_rw_m[:,0:1,:].float().to(device),p_rw_d[:,0:1,:].float().to(device)],dim=1),
                          torch.cat([n_x.detach(),r_x.detach()],dim=1))
            l_m.append(label.float())
            p_m.append(output.cpu().detach())
    test_metric.append(caculate_metrics(torch.cat(p_m,dim=0),torch.cat(l_m,dim=0),'softmax'))