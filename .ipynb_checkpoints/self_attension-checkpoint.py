from math import sqrt
import torch
import torch.nn as nn
import pdb
import config as cfg
from vgtk.so3conv import functional as L
from SPConvNets.options import opt as Eopt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        self.mlp = ElementWiseMLP()
        
    def forward(self, x):
        B,_,_,_ = x.shape
        out_bach = torch.empty(size=(B, cfg.NUM_PARTS,60), device=x.device)
        for i in range(B):
            queries = self.query(x[i])
            keys = self.key(x[i])
            values = self.value(x[i])
            scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
            attention = self.softmax(scores)
            weighted = torch.bmm(attention, values)
            output = self.mlp(weighted).squeeze(2)
            out_bach[i] = output
        return out_bach

class ElementWiseMLP(nn.Module):
    def __init__(self):
        super(ElementWiseMLP, self).__init__()
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x
    
    
# input = torch.rand(16,10, 60, 64).to(device)
# model= SelfAttention(64).to(device)
# weight = model(input)
# B = weight.shape[0]

def Chordal_L2(weight):
    B = weight.shape[0]
    rotamatrix_all = torch.empty(size=(B, cfg.NUM_PARTS, 3,3), device=device)
    for i in range(B):
        weight_p = weight[i] / weight[i].sum(dim=1, keepdim=True)
        anchors = torch.tensor(L.get_anchors(60)).to(device)
        Rj = torch.einsum('ij,jkl->ikl', weight_p, anchors)
        Rk_all = torch.empty(size=(cfg.NUM_PARTS, 3,3), device=device)
        for j in range(8):
            U,D,V = torch.svd(Rj[j])
            if torch.linalg.det(torch.matmul(U,V.T)) >=0:
                Rk = torch.matmul(U,V.T)
            else:
                diag = torch.tensor([[1.,0.,0.],
                                    [0.,1.,0.],
                                    [0.,0.,-1.]]).to(device)
                Rk = U @ diag @ V.T
            Rk_all[j] = Rk
        rotamatrix_all[i] = Rk_all
    return rotamatrix_all


    
    
