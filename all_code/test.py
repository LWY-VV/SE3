import torch
import trimesh
import pybullet as p
import pybullet as pb
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
    DifferentiableTwoLinkRobot,
    DifferentiableFrankaPanda
)
# from pytorch3d.transforms import quaternion_to_matrix
from util import quaternion_to_matrix
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
# from util import pandaPC_generate, pc_visualize, get_theta, get_local_rotamatrix,panda_forward_kinematic
from util import *
from SE3_model import EPNGeM
# from util import test_pc_generate

######### model initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Eopt.device = device
Eopt.model.kanchor = 60
net = EPNGeM(Eopt).to(device)                                                                                                                                                                                                                     
net.load_state_dict(torch.load('./checkpoint_quat/099.pth'))
#########

# pcs = np.vstack([panda_pc_1,panda_pc_2])
data_overfit = np.load('./panda_data/se3_data_test/3.npz')
pc_of = torch.tensor(np.expand_dims(data_overfit['pointcloud'], axis=0)[:,::2]).float().to(device)
RT_of = torch.tensor(np.expand_dims(data_overfit['GT'], axis=0)).float().to(device)
label = torch.tensor(np.expand_dims(data_overfit['label'], axis=0)[:,::2]).to(device)
joints = torch.tensor(data_overfit['Joints']).to(device)
ee_pos = torch.tensor(data_overfit['EE_pos']).to(device)
# {"pointcloud": pointcloud, "GT": rotat_matrix_all, "label": label}
pre_rotmatrx, quaternion_all = net(pc_of,label)


# joints_pre = rotatrix2jointstate(pre_rotmatrx)
# pdb.set_trace()
# pc_test = test_pc_generate(pre_rotmatrx, joint).detach().cpu().numpy()
local_rotmatrix = get_local_rotamatrix(pre_rotmatrx).reshape(7,3,3)



### subspace projection
matrix_after_project_all = []
for i in range(local_rotmatrix.shape[0]):
    local_rotmatrix_notheta = torch.tensor(np.load('rotmatrix_notheta.npy')).float().cuda()
    onlytheta = local_rotmatrix[i] @ torch.linalg.inv(local_rotmatrix_notheta[i])
    quaternion = matrix_to_quaternion(onlytheta)
    if i == 0:
        project = torch.dot(quaternion[1:], torch.tensor([0.,0.,1.]).cuda())
#         new_quat = torch.cat([quaternion[0], 0., 0., project]).cuda()
        quaternion[1] = 0
        quaternion[2] = 0
        quaternion[3] = project
        
    else:
        project = torch.dot(quaternion[1:], torch.tensor([0.,1,0.]).cuda())
#         new_quat = torch.cat([quaternion[0], 0., 0., project]).cuda()
        quaternion[1] = 0
        quaternion[3] = 0
        quaternion[2] = project
        
    matrix_after_projection = quaternion_to_matrix(quaternion)
    matrix_after_project_all.append(matrix_after_projection)

matrix_after_project_all = torch.vstack(matrix_after_project_all).reshape(7,3,3)


local_rot_all = []
local_rot_all.append(pre_rotmatrx[0][0])
for i in range(7):
    local_rot_all.append(matrix_after_project_all[i] @ local_rotmatrix_notheta[i])

local_rot_all = torch.vstack(local_rot_all).reshape(8,3,3)

    

pc_test = panda_forward_kinematic_withlocalmatrix(local_rot_all).detach().cpu().numpy()


# pc_test = panda_forward_kinematic(pre_rotmatrx).detach().cpu().numpy()
pc_gt = pc_of[0].detach().cpu().numpy()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs, ys, zs = pc_test[:, 0], pc_test[:, 1], pc_test[:, 2]
xg, yg, zg = pc_gt[:, 0], pc_gt[:, 1], pc_gt[:, 2]
# xs, ys, zs =0.,0.,0.


ax.scatter(xs, ys, zs,c='b', marker='o',s=2)

ax.scatter(xg, yg, zg,c='r', marker='o',s=2)

# ax.scatter(x_ax,y_ax,z_ax, c='r', marker='o')


ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=20, azim=30)
# plt.savefig(str(filename)+'.png')
plt.show()
