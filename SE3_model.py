"""
network architechture 
"""
import torch
import torch.nn as nn
# import vgtk
# import SPConvNets.utils as M
# import vgtk.spconv.functional as L

import SPConvNets.models.pr_so3net_pn as frontend

import config as cfg
from SPConvNets.options import opt as Eopt
import pdb
# from pointnet.pointnet_changeDim import PointNetDenseCls, STNkd
from self_attension import SelfAttention
from self_attension import Chordal_L2
from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Eopt.device = device
Eopt.model.kanchor = 60
class EPNGeM(nn.Module):
    def __init__(self, opt):
        super(EPNGeM, self).__init__()
        self.opt = opt
        # epn param
        self.mlps=[[64]]
        out_mlps=[self.mlps[-1][0], cfg.LOCAL_FEATURE_DIM]
        self.epn = frontend.build_model(self.opt, self.mlps, out_mlps, outblock='linear')
        
        ### pointnet
        # self.pointnet = PointNetDenseCls(k=8,feature_transform=True)
        self.selfattension = SelfAttention(64)

    def forward(self, x, label):
        '''
        INPUT: B, N, D_input=3
        Local Feature: B, N', D_local
        Global Feature: B, D_output
        '''
        B, point_num, _ = x.shape
        # x_invariant = torch.empty(size=(B, point_num, cfg.LOCAL_FEATURE_DIM), device=x.device)
        x_equvariant = torch.empty(size=(B,cfg.LOCAL_FEATURE_DIM, point_num, Eopt.model.kanchor), device=x.device)
        
        for i in range(B):
            x_onlyone = x[i, :, :].unsqueeze(0)# 1*number_of_points*4
            # x_invariant[i], x_equvariant[i] = self.epn(x_onlyone)# b*np*outdim(invariant feature) 6*500*1024
            _, x_equvariant[i] = self.epn(x_onlyone)
        x_equvariant = x_equvariant.permute(0,2,3,1)
        
        
        aggregated_features = torch.empty(B, cfg.NUM_PARTS, 60, 64).to(device)
        
        for i in range(1,9):
            # mask = (label == i).unsqueeze(-1).float()
            mask = torch.zeros(B, point_num, 60, 64).to(device)
            idx = torch.where(label == i)[1]
            # idx = x_equvariant[torch.where(label == i)]
            mask[:,idx,:,:] = 1
            masked_features = x_equvariant * mask
            averaged_features = masked_features.sum(1) / mask.sum(1)
            aggregated_features[:, i-1, :, :] = averaged_features
        rotamatrix_all_weight = self.selfattension(aggregated_features)
        rotmatrix = Chordal_L2(rotamatrix_all_weight)
        ##### rotmatrix is the predicted global rotation matrix
        local_rotmatrix = get_local_rotamatrix(rotmatrix).reshape(7,3,3)
        
        
        quaternion_all = []
        for i in range(local_rotmatrix.shape[0]):
            local_rotmatrix_notheta = torch.tensor(np.load('rotmatrix_notheta.npy')).float().to(device)
            onlytheta = local_rotmatrix[i] @ torch.linalg.inv(local_rotmatrix_notheta[i])
            quaternion = matrix_to_quaternion(onlytheta)
            if i == 0:
                project = torch.dot(quaternion[1:], torch.tensor([0.,0.,1.]).to(device))
                quaternion[1] = 0
                quaternion[2] = 0
                quaternion[3] = project

            else:
                project = torch.dot(quaternion[1:], torch.tensor([0.,1,0.]).to(device))
                quaternion[1] = 0
                quaternion[3] = 0
                quaternion[2] = project

#             matrix_after_projection = quaternion_to_matrix(quaternion)
#             matrix_after_project_all.append(matrix_after_projection)
            quaternion_all.append(quaternion)
        quaternion_all = torch.vstack(quaternion_all)
        return rotmatrix, quaternion_all
        
# labels = torch.randint(0, 10, (16, 500)).to(device)
# input = torch.rand(16,cfg.NUM_POINTS,3).to(device)
# model = EPNGeM(Eopt,labels).to(device)
# norm_weight  = model(input)
# # rotmatrix = Chordal_L2(norm_weight)
# pdb.set_trace()


