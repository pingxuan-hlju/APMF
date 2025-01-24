import torch
import torch.nn as nn
import numpy as np
import random
import os
import math
from sklearn import metrics
from torch_cluster import random_walk
from torch.utils.data import dataset,dataloader

seed=1
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
common_set=torch.load('./data/train_data/common_set.pkl')
train_set=torch.load('./data/train_data/train_set.pkl')
test_set=torch.load('./data/train_data/test_set.pkl')

kfolds=5
batch=2048       
adj_threshold=0.9
nce_temp=1e+2   
calpha=1.0
input_dim=common_set['md'].shape[0]+common_set['md'].shape[1]+common_set['ml'].shape[1]
rw_param={'rw_len':5,'rw_times':20,'rw_p':1,'rw_q':1}    
C_model_param={'n_heads':2,     
               'hidden_size':512,    
               'lr':9e-5,            
               'weight_decay':5e-4,   
               'epochs':100}          
P_model_param={'hidden_channle':8,   
               'num_group':2,      
               'gate_treshold':0.5,   
               'dropout_rate':0.2,  
               'lr':7e-5,           
               'weight_decay':5e-4,   
               'epochs':100}          

device=torch.device('cuda')

class MyDataset(dataset.Dataset):
    def __init__(self, edges, labels, fea, prw, nrw, nm):
        self.nm=nm
        self.Data = edges
        self.Label = labels
        self.fea=fea
        self.prw=prw
        self.nrw=nrw
    def __len__(self):
        return len(self.Label)
    def __getitem__(self, index):
        xy = self.Data[index]
        label = self.Label[index]
        p_rw_m=self.fea[self.prw[xy[0]]]
        p_rw_d=self.fea[self.prw[xy[1]+self.nm]]
        n_rw_m=self.fea[self.nrw[xy[0]]]
        n_rw_d=self.fea[self.nrw[xy[1]+self.nm]]
        return xy, label, p_rw_m, p_rw_d, n_rw_m, n_rw_d

    def __init__(self, kfolds, rw_param, adj_threshold, common_set, train_set, test_set):
        self.nm=common_set['md'].shape[0]
        self.rw_param=rw_param
        self.kfolds=kfolds
        self.adj_threshold=adj_threshold
        self.feas,self.prws,self.nrws=[],[],[]
    def negative_sampling(self,fea,prw):
        for j in range(self.rw_param['rw_len']):
            fea[prw[:,0],prw[:,j+1]]=0
        return self.RandomWalk(fea)
    def RandomWalk(self,fea):
        adj=torch.argwhere(fea>self.adj_threshold)
        rw_nodes=random_walk(adj[:,0],adj[:,1],torch.arange(fea.shape[0]).repeat_interleave(self.rw_param['rw_times']),
                                walk_length=self.rw_param['rw_len'],p=self.rw_param['rw_p'],q=self.rw_param['rw_q'])
        rws=[]
        for j in range(fea.shape[0]):
            rws.append([j])
            for m in range(self.rw_param['rw_len']):
                uv,co=torch.unique(rw_nodes[j*self.rw_param['rw_times']:(j+1)*self.rw_param['rw_times'],m],return_counts=True)
                rws[j].append(uv[co.argmax()].item())
        return torch.tensor(rws).long()
    def tr_va_te_data_set(self, edge, label, batch, i):
        return dataloader.DataLoader(MyDataset(edge, label, self.feas[i], self.prws[i], self.nrws[i], self.nm), batch_size=batch, shuffle=True, num_workers=0)
class DataProcess():
    def __init__(self, kfolds, rw_param, adj_threshold, common_set, train_set, test_set):
        self.nm=common_set['md'].shape[0]
        self.rw_param=rw_param
        self.kfolds=kfolds
        self.adj_threshold=adj_threshold
        self.feas,self.prws,self.nrws=[],[],[]
        fea=torch.cat([torch.cat([test_set['mm_mdF'],test_set['md']],dim=1),
                        torch.cat([test_set['md'].t(),common_set['dd_sem']],dim=1)],dim=0)
        self.prws.append(self.RandomWalk(fea))
        self.nrws.append(self.negative_sampling(fea,self.prws[-1]))
        fea=torch.cat([fea,torch.cat([common_set['ml'],common_set['dl']],dim=0)],dim=1)
        self.feas.append(fea)
    def negative_sampling(self,fea,prw):
        for j in range(self.rw_param['rw_len']):
            fea[prw[:,0],prw[:,j+1]]=0
        return self.RandomWalk(fea)
    def RandomWalk(self,fea):
        adj=torch.argwhere(fea>self.adj_threshold)
        rw_nodes=random_walk(adj[:,0],adj[:,1],torch.arange(fea.shape[0]).repeat_interleave(self.rw_param['rw_times']),
                                walk_length=self.rw_param['rw_len'],p=self.rw_param['rw_p'],q=self.rw_param['rw_q'])
        rws=[]
        for j in range(fea.shape[0]):
            rws.append([j])
            for m in range(self.rw_param['rw_len']):
                uv,co=torch.unique(rw_nodes[j*self.rw_param['rw_times']:(j+1)*self.rw_param['rw_times'],m],return_counts=True)
                rws[j].append(uv[co.argmax()].item())
        return torch.tensor(rws).long()
    def tr_va_te_data_set(self, edge, label, batch, i):
        return dataloader.DataLoader(MyDataset(edge, label, self.feas[i], self.prws[i], self.nrws[i], self.nm), batch_size=batch, shuffle=True, num_workers=0)
def caculate_metrics(p_m,l_m,act_type='softmax'):
    if act_type=='softmax':
        fl_p=torch.softmax(p_m,dim=1)[:,1].numpy()
        ol_p=torch.argmax(p_m,dim=1).numpy()
    elif act_type=='sigmoid':
        fl_p=p_m.numpy()
        ol_p=(p_m>0.5).long().numpy()
    l_m=l_m.numpy()
    p,r,_= metrics.precision_recall_curve(l_m,fl_p)
    metric_result=[metrics.roc_auc_score(l_m,fl_p),metrics.auc(r, p),metrics.accuracy_score(l_m,ol_p),
                                    metrics.f1_score(l_m,ol_p),metrics.recall_score(l_m,ol_p)]
    print("auc:"+str(metric_result[0])+";aupr:"+str(metric_result[1])+";accuracy:"+str(metric_result[2])+";f1_score:"+str(metric_result[3])+";recall:"+str(metric_result[4]))
    return metric_result
class EarlyStopping:
    def __init__(self,savepath,patience=3,delta=0):  
        self.savepath=savepath
        self.patience=patience
        self.bestscore=None
        self.delta=delta
        self.counter=0
        self.earlystop=False
    def __call__(self,score,model):
        fscore=-score
        if self.bestscore is None:
            self.bestscore=fscore
            torch.save(model.state_dict(),self.savepath)
        elif fscore<self.bestscore+self.delta:
            self.counter+=1
            if self.counter>=self.patience:
                self.earlystop=True
        else:
            self.bestscore=fscore
            torch.save(model.state_dict(),self.savepath)
            self.counter=0
def info_NCE_loss(p_output,n_output,nce_temp):
        zd=torch.exp(p_output[:,0:1,:]@n_output.transpose(1,2)/nce_temp)
        sm=-torch.log(zd[:,0,0]+1e-8)
        xm=torch.log(zd[:,0,:].sum(-1)+1e-8)
        return (sm+xm).mean()
def contrastive_loss(p_n_output,p_r_output,n_n_output,n_r_output,nce_temp,calpha):
        p_n_m,p_n_d=torch.split(p_n_output,p_n_output.shape[1]//2,dim=1)
        n_n_m,n_n_d=torch.split(n_n_output,n_n_output.shape[1]//2,dim=1)
        node_loss=info_NCE_loss(p_n_m,n_n_m,nce_temp)+info_NCE_loss(p_n_d,n_n_d,nce_temp)
        relation_loss=(p_r_output*n_r_output).sum(-1)/(torch.sqrt((p_r_output**2).sum(-1))*torch.sqrt((n_r_output**2).sum(-1)))
        relation_loss=(relation_loss+1)/2
        return node_loss+calpha*relation_loss.mean()

class CNet(nn.Module):
    def __init__(self,input_dim,hidden_dim,heads,walk_len):
        super(CNet,self).__init__()
        self.ml1,self.dl1=nn.Linear(input_dim,hidden_dim),nn.Linear(input_dim,hidden_dim)
        self.mn1,self.dn1=nn.LayerNorm(hidden_dim),nn.LayerNorm(hidden_dim)
        self.LMHA=nn.MultiheadAttention(hidden_dim,heads,batch_first=True)
        self.Ln=nn.LayerNorm(hidden_dim)
        self.ml2,self.dl2=nn.Linear(walk_len,1),nn.Linear(walk_len,1)
        self.mn2,self.dn2=nn.LayerNorm(hidden_dim),nn.LayerNorm(hidden_dim)
        self.GMHA=nn.MultiheadAttention(hidden_dim,heads,batch_first=True)
        self.Gn=nn.LayerNorm(hidden_dim)
        self.reset_parameters()
    def forward(self,xm,xd):
        xm,xd=self.mn1(self.ml1(xm)),self.dn1(self.dl1(xd))
        xmd=torch.cat([xm,xd],dim=1)
        nx=self.Ln(self.LMHA(xmd,xmd,xmd)[0])
        xm=self.mn2(self.ml2(xm.transpose(1,2)).transpose(1,2))
        xd=self.dn2(self.dl2(xd.transpose(1,2)).transpose(1,2))
        xmd=torch.cat([xm,xd],dim=1)
        rx=self.Gn(self.GMHA(xmd,xmd,xmd)[0])
        return nx,rx

class PNet(nn.Module):
    def __init__(self,input_dim,hidden_dim,hidden_channle,num_group,gate_treshold=0.5,dropout_rate=0):
        super(PNet,self).__init__()
        self.proj1=nn.Linear(input_dim,hidden_dim)
        self.proj2=nn.Linear(hidden_dim,hidden_dim)
        self.norm=nn.LayerNorm(hidden_dim)
        self.conv1=nn.Conv2d(1,hidden_channle,kernel_size=(1,1),stride=1,padding=0)
        self.gn=nn.GroupNorm(num_groups=num_group,num_channels=hidden_channle)
        self.gate_treshold=gate_treshold
        self.gc=nn.Conv2d(hidden_channle,num_group,kernel_size=(2,7),stride=(2,7),padding=0,groups=num_group)
        self.pool=nn.AdaptiveAvgPool2d((16,512))
        self.l1=nn.Linear(num_group*1024,512)
        self.dropout=nn.Dropout(dropout_rate)
        self.l2=nn.Linear(512,2)
        self.sigmoid=nn.Sigmoid()
        self.leakyrelu=nn.LeakyReLU()
        self.reset_parameters()
    def forward(self,x1,x2):
        x=torch.cat([self.proj1(x1),self.proj2(x2)],dim=0)
        x=self.norm(x)
        x=self.leakyrelu(self.conv1(x[:,None,:,:]))
        gn_x=self.gn(x)
        w_gamma=self.gn.weight/sum(self.gn.weight)
        reweight=self.sigmoid(gn_x*w_gamma[None,:,None,None])
        x1=torch.where(reweight>self.gate_treshold,torch.ones_like(reweight),reweight)*x
        x2=torch.where(reweight<self.gate_treshold,torch.zeros_like(reweight),reweight)*x
        x11,x12=torch.split(x1,x1.size(1)//2,dim=1)
        x21,x22=torch.split(x2,x2.size(1)//2,dim=1)
        x=torch.cat([x11+x22,x12+x21],dim=1)
        x=self.pool(self.leakyrelu(self.gc(x)))
        return self.l2(self.dropout(self.l1(x.reshape(x.shape[0],-1))))

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
