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
from util import pandaPC_generate, pc_visualize, get_local_rotamatrix
from SE3_model import EPNGeM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_overfit = np.load('./panda_data/se3_data_test/17.npz')
pc_of = torch.tensor(np.expand_dims(data_overfit['pointcloud'], axis=0)[:,::2]).float().to(device)
RT_of = torch.tensor(np.expand_dims(data_overfit['GT'], axis=0)).float().to(device)
label = torch.tensor(np.expand_dims(data_overfit['label'], axis=0)[:,::2]).to(device)
joints = torch.tensor(data_overfit['Joints'])


##### code that generate the rotation matrix without theta
def get_tf_mat_notheta(i, dh):
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    q = theta

    return np.array([[1., 1., 1.],
                     [np.cos(alpha), np.cos(alpha), -np.sin(alpha)],
                     [np.sin(alpha), np.sin(alpha), np.cos(alpha)]
                     ])

def get_tf_mat(i, dh):
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    q = theta

    return np.array([[np.cos(q), -np.sin(q), 0],
                     [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha)],
                     [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha)]
                     ])



def get_Transfrom(joint_angles):
    dh_params = np.array([[0, 0.333, 0, joint_angles[0]],
                 [0, 0, -np.pi / 2, joint_angles[1]],
                 [0, 0.316, np.pi / 2, joint_angles[2]],
                 [0.0825, 0, np.pi / 2, joint_angles[3]],
                 [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
                 [0, 0, np.pi / 2, joint_angles[5]],
                 [0.088, 0, np.pi / 2, joint_angles[6]],
                 [0, 0.107, 0, 0],
                 [0, 0, 0, -np.pi / 4],
                 [0.0, 0.1034, 0, 0]], dtype=np.float64)

    T_all = []
    for i in range(7):
        T_all.append(get_tf_mat(i, dh_params))
    T_all = np.array(T_all)
    return T_all

def rotateY(matrix):

            qr, _ = np.linalg.qr(matrix)  # Orthogonalize using QR decomposition
            # Define the target axis of rotation (y-axis)
            target_axis = np.array([0, 1, 0])
            # Compute rotation angle
            cos_theta = np.dot(qr[:, 0], target_axis)
            pdb.set_trace()
            sin_theta = np.linalg.norm(np.cross(q[:, 0], target_axis))
            rotation_angle = np.arctan2(sin_theta, cos_theta)

            # Construct the rotation matrix
            rotation_matrix = np.array([[np.cos(rotation_angle), 0,  np.sin(rotation_angle)],
                                        [0, 1, 0],
                                        [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]])

            # Project the orthogonalized matrix onto the rotation matrix
            constrained_matrix = q.dot(rotation_matrix)
            return constrained_matrix


RT_of = RT_of[:,:-1,:,:]
joints = joints.reshape(1,-1)[0]
local_rotmatrix = get_local_rotamatrix(RT_of).reshape(7,3,3)

for i in range(7):
    local_rotmatrix_notheta = torch.tensor(np.load('rotmatrix_notheta.npy')).float().cuda()
    onlytheta = local_rotmatrix[i] @ torch.linalg.inv(local_rotmatrix_notheta)[i]
    qr, _ = torch.linalg.qr(onlytheta)
    pdb.set_trace()


rot_matrix = local_rotmatrix[3]
pdb.set_trace()
rotateY(rot_matrix)
# joints = torch.zeros(7)
# local_rotmatrix_notheta = torch.tensor(get_Transfrom(joints))

# onlytheta = local_rotmatrix @ torch.linalg.inv(local_rotmatrix_notheta)
# pdb.set_trace()
# T_DH_nontheta = get_Transfrom(joint)
# np.save("rotmatrix_notheta", local_rotmatrix_notheta)

#         theta_pre = torch.vstack(get_theta(local_rotmatrix)).reshape(joints.shape[0], joints.shape[1])
        #####

#         test_pc = panda_forward_kinematic(pred_rotmatrix)
#         import pdb ; pdb.set_trace()

#         gt_pc, gt_ee = pandaPC_generate(joints)
#         test_pc2, test_ee = pandaPC_generate(joints)
#         gt_pc = torch.tensor(gt_pc)[:4500,:-1].float().cuda()
#         test_pc2 = torch.tensor(test_pc2)[:4500,:-1].float().cuda()

