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