import torch
import torch.optim as optim
from SE3_model import EPNGeM
import numpy as np
from torch.utils.data import DataLoader
from dataset import pandajoints_Dataset
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from vgtk.utils import *
import torch.nn.utils as utils
import os
import pdb
from SPConvNets.options import opt as Eopt
from tqdm import tqdm
# from util import Chordal_distance, pandaPC_generate, panda_forward_kinematic,get_local_rotamatrix, get_theta, EE_position, get_EE_pos, matrix_to_quaternion
from util import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Eopt.device = device
Eopt.model.kanchor = 60

train_dataset, test_dataset = pandajoints_Dataset(train=True), pandajoints_Dataset(train=False)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

epochs = 100
net = EPNGeM(Eopt).to(device)
opt = optim.Adam(net.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=len(train_dataloader)*3, gamma=0.3)
logdir = 'logs/test_quat'
for f in os.listdir(logdir):
    os.remove(os.path.join(logdir, f))
writer = SummaryWriter(logdir)
mse = torch.nn.MSELoss()


# def test(test_dataloader, net, writer, step):
#     net.eval()
#     error = 0
#     with torch.no_grad():
#         for data in test_dataloader:
#             pc, gt, label = data[:,::18]
#             pc = pc.float().to(device)
#             gt = gt.float().to(device)
#             label = label.to(device)[:,::18]
#             pred_rotmatrix = net(pc,label)
#             # error += mse(pred_rotmatrix, gt)
#             error += Chordal_distance(pred_rotmatrix, gt)
    
#     writer.add_scalar('test_error', error/len(test_dataloader), global_step=step)
#     net.train()
    
step = 0
for epoch in range(epochs):
    for i, data in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
        pc, gt, label, joints, ee_pos, quaternion = data
        
        # get one point from every two points, reduce the GPU cost
        pc_train = pc.float().to(device)[:,::2]
#         gt = gt[:,:-1,:,:].float().to(device)
        gt = gt.float().to(device)
        label = label.to(device)[:,::2]
        joints = joints.reshape(1,-1).to(device)
        ee_pos = ee_pos.squeeze().squeeze().to(device)
        quaternion_gt = quaternion.squeeze().float().to(device)
        pred_rotmatrix, quaternion_pre = net(pc_train,label)

        # lossquater = mse(pred_rotmatrix, gt)
        loss_matrix = Chordal_distance(pred_rotmatrix, gt)
        loss_quat = mse(quaternion_pre, quaternion_gt)
        
        
#         ee_test = get_EE_pos(pred_rotmatrix)
#         loss_ee = mse(ee_test, ee_pos)
        loss = loss_matrix + loss_quat 
        loss.backward()
        
        opt.step()
        # scheduler.step()
        opt.zero_grad()
        step += 1

        if step % 10 == 0:
            writer.add_scalar('train_loss', loss, global_step=step)
        # if step % 50 == 0:
        #     pc1, pred_pc2, pc2 = dcn(pc1).permute(0, 2, 1), dcn(pred_pc2).permute(0, 2, 1), dcn(pc2).permute(0, 2, 1)
        #     writer.add_mesh('point_cloud', vertices=torch.concat([pc1, pred_pc2, pc2], axis=1), colors=torch.concat([c1, c_pred, c2], axis=1), global_step=step)
        if step % 1000 == 0:
            torch.save(net.state_dict(), f'checkpoint_quat/{str(epoch).zfill(3)}.pth')