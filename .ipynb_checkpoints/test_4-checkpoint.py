import torch
import trimesh
import pybullet as p
import pybullet as pb
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
    DifferentiableTwoLinkRobot,
    DifferentiableFrankaPanda
)

import pybullet_data
import numpy as np
# import mesh_to_sdf
# from mesh_to_sdf import sample_sdf_near_surface
# from SDF import compute_unit_sphere_transform
import matplotlib.pyplot as plt
import pdb
# import rospy
# from sensor_msgs.msg import JointState

from SPConvNets.options import opt as Eopt
from util import pandaPC_generate, pc_visualize, panda_forward_kinematic,transformation_matrix
from SE3_model import EPNGeM
from util import test_pc_generate
import numpy.matlib as matlib
from math import sin, cos, atan2, sqrt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Eopt.device = device
Eopt.model.kanchor = 60
net = EPNGeM(Eopt).to(device)                                                                                                                                                                                                                     
net.load_state_dict(torch.load('./checkpoint_theta/099.pth'))


data_overfit = np.load('./panda_data/se3_data/190.npz')
pc_of = torch.tensor(np.expand_dims(data_overfit['pointcloud'], axis=0)[:,::2]).float().to(device)
RT_of = torch.tensor(np.expand_dims(data_overfit['GT'], axis=0)).float().to(device)
label = torch.tensor(np.expand_dims(data_overfit['label'], axis=0)[:,::2]).to(device)
joints = torch.tensor(data_overfit['Joints']).to(device)
ee_pos = torch.tensor(data_overfit['EE_pos']).to(device)



def get_EE_pos(pre_rotmatrix):
    pre_rotmatrix = pre_rotmatrix[0]
    sample_point_num = 500
    rot_local_all = []
    rot_local_all.append(pre_rotmatrix[0])
    
    for i in range(pre_rotmatrix.shape[0]-1):
        rot_local = pre_rotmatrix[i].T @ pre_rotmatrix[i+1]
        rot_local_all.append(rot_local)
#     pdb.set_trace()
#     rot_local_all = torch.tensor(rot_local_all)
    offsets = torch.tensor([[0., 0., 0.333],
                        [0.,   0.,  0.],
                        [0, -0.316, 0.],
                        [0.0825,0., 0.],
                        [-0.0825, 0.384, 0],
                        [0., 0., 0.],
                        [0.088, 0., 0.],
                        [0.,0.,0.107]]).float().cuda()
    Translation_all = []
    for i in range(len(offsets)-1):

        m = transformation_matrix(rot_local_all[i+1],offsets[i])
        Translation_all.append(m)

    trans_hand = transformation_matrix(rot_local_all[-1],offsets[-1])
    Translation_all.append(trans_hand)
    
    final = Translation_all[0]@ Translation_all[1]@ Translation_all[2] @ Translation_all[3] @ Translation_all[4]@ Translation_all[5] @ Translation_all[6] @ Translation_all[7]
    hand_pos = torch.tensor([final[0,3] ,final[1,3], final[2,3]])
    return hand_pos
Rt = RT_of[:,:-1,:,:]
ee = get_EE_pos(Rt)
ee_gt = ee_pos
pdb.set_trace()
# ######### model initialize
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Eopt.device = device
# Eopt.model.kanchor = 60
# net = EPNGeM(Eopt).to(device)                                                                                                                                                                                                                     
# net.load_state_dict(torch.load('./checkpoint_forward_kine/099.pth'))
# #########

# # pcs = np.vstack([panda_pc_1,panda_pc_2])
# data_overfit = np.load('./panda_data/se3_data_test/12.npz')
# pc_of = torch.tensor(np.expand_dims(data_overfit['pointcloud'], axis=0)[:,::2]).float().to(device)
# RT_of = torch.tensor(np.expand_dims(data_overfit['GT'], axis=0)).float().to(device)
# label = torch.tensor(np.expand_dims(data_overfit['label'], axis=0)[:,::2]).to(device)
# joint = data_overfit['Joints']
# # {"pointcloud": pointcloud, "GT": rotat_matrix_all, "label": label}
# pre_rotmatrix = net(pc_of,label)

# pc_gt = pandaPC_generate(joint)
# pc_test = panda_forward_kinematic(pre_rotmatrix).cpu().detach().numpy()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# xs, ys, zs = pc_test[:, 0], pc_test[:, 1], pc_test[:, 2]
# xg, yg, zg = pc_gt[:, 0], pc_gt[:, 1], pc_gt[:, 2]
# # xs, ys, zs =0.,0.,0.


# ax.scatter(xs, ys, zs,c='b', marker='o',s=2)

# ax.scatter(xg, yg, zg,c='r', marker='o',s=2)

# # ax.scatter(x_ax,y_ax,z_ax, c='r', marker='o')


# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([0, 1.2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.view_init(elev=20, azim=30)
# # plt.savefig(str(filename)+'.png')
# plt.show()

# #