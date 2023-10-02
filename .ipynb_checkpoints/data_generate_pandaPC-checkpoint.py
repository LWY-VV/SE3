import torch
import trimesh
import pybullet as p
import pybullet as pb
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
    DifferentiableTwoLinkRobot,
)
from util import quaternion_to_matrix
import pybullet_data
import numpy as np
# import mesh_to_sdf
# from mesh_to_sdf import sample_sdf_near_surface
# from SDF import compute_unit_sphere_transform
import matplotlib.pyplot as plt
import pdb
from transforms3d.euler import mat2euler, euler2quat
# import rospy
# from sensor_msgs.msg import JointState

begin_state_all = []
test_case_number = 500
joints_all = []
# test dataset
# np.random.seed(36)
# train dataset
np.random.seed(20)

for i in range(1000):
    """
    generate random 1000 poses of panda arm
    
    output:
    
    pointcloud: 4 dimension, first 3 are coordinates, the last is label(which link this point belong to)
    "GT": groundtruth rotat_matrix_all, 
    "label": label, 
    "Joints": joint angles of panda arm  
    "EE_pos": ee_position,
    "quaternion":quat_all} 
    """
    
    jointvalue1 = np.random.uniform(-2.9, 2.9) 
    jointvalue2 = np.random.uniform(-1.8, 1.8)
    jointvalue3 = np.random.uniform(-2.9, 2.9)
    jointvalue4 = np.random.uniform(-3.14, 0.08)
    jointvalue5 = np.random.uniform(-2.9, 2.9)
    jointvalue6 = np.random.uniform(-0.08, 3.8)
    jointvalue7 = np.random.uniform(-2.9, 2.9)

# for i in range(50):
#     jointvalue1 = np.random.uniform(-1.5, 1.5) 
#     jointvalue2 = np.random.uniform(-1.5, 1.5)
#     jointvalue3 = np.random.uniform(-1.5, 1.5)
#     jointvalue4 = np.random.uniform(-1.5, 0.08)
#     jointvalue5 = np.random.uniform(-1.5, 1.5)
#     jointvalue6 = np.random.uniform(-0.08, 1.5)
#     jointvalue7 = np.random.uniform(-1.5, 1.5)
    
    joint_all = np.array([jointvalue1,jointvalue2,jointvalue3,jointvalue4,jointvalue5,jointvalue6,jointvalue7]).reshape(1,7)
    # begin_state_all.append(joint_all)
    joints_all.append(joint_all)
# begin_state_all = np.array(begin_state_all).reshape(test_case_number,7)


    mesh_path_panda = './panda/panda.urdf'
    panda_robot = DifferentiableRobotModel(urdf_path=mesh_path_panda, name="panda", device='cuda')
    panda_q = torch.tensor(joint_all).cuda()
    panda_links_all = panda_robot.compute_forward_kinematics_all_links(q=panda_q)
    points_all = []
    link_idx = 1
    rotat_matrix_all = []
    for link_name in panda_links_all:
        
        if link_name == 'world'or link_name == 'panda_link8' or link_name == 'panda_leftfinger'or link_name == 'panda_rightfinger':
                continue
        new_string = link_name.replace("panda_", "")
        path2_mesh = './panda/meshes/visual_stl/' + new_string +'.stl'
        
        mesh = trimesh.load(path2_mesh, force='mesh')
        pc = trimesh.sample.sample_surface(mesh,1000)[0]
              
        # new_string = link_name.replace("panda_", "")
        # path2_mesh = '/home/wangyi/se3_equivariant_place_recognition-master/panda/meshes/visual/' + new_string +'.dae'
        # mesh = trimesh.load(path2_mesh, force='mesh')
        # pc = mesh.vertices
        # pdb.set_trace()

        orientation = panda_links_all[link_name][1][0]
        link_orientation = torch.cat((orientation[-1:], orientation[:-1]))
        rotat_matrix = quaternion_to_matrix(link_orientation).cpu().tolist()
        rotat_matrix_all.append(rotat_matrix)
        link_position = panda_links_all[link_name][0][0].cpu().tolist()
        point_world = (rotat_matrix @ pc.T).T + link_position
        
        idxs = np.ones((pc.shape[0], 1))*link_idx
        pc_with_idx = np.concatenate((point_world, idxs), axis=1)
        
        points_all.append(pc_with_idx)
        link_idx += 1
        if link_idx == 9:
            link_idx = 8
    points_all = np.vstack(points_all)
    rotat_matrix_all = np.array(rotat_matrix_all)[:-1,:,:]

    # generate quaternion
    rot_local_all = []
    for j in range(rotat_matrix_all.shape[0]-1):
        rot_local = rotat_matrix_all[j].T @ rotat_matrix_all[j+1]
        rot_local_all.append(rot_local)
    rot_local_all = np.vstack(rot_local_all).reshape(7,3,3)
    non_theta = np.load('rotmatrix_notheta.npy')
    quat_all = []
    for k in range(rot_local_all.shape[0]):
        onlytheta = rot_local_all[k] @ np.linalg.inv(non_theta[k])
        euler = mat2euler(onlytheta)
        quat = euler2quat(euler[0],euler[1],euler[2])
        quat_all.append(quat)
    quat_all = np.array(quat_all)
   
    pointcloud = points_all[:,:3]
    label = points_all[:,-1:]
    ee_position = panda_links_all['panda_hand'][0].detach().cpu().numpy()

    save_data = {"pointcloud": pointcloud, "GT": rotat_matrix_all, "label": label, "Joints": joint_all, "EE_pos": ee_position,
                "quaternion":quat_all}
    
    path = './panda_data/se3_quaternion/'
    savepath = path + str(i) + '.npz'
    np.savez(savepath, **save_data)
# joints_all = np.array(joints_all)
# pdb.set_trace()
    # pdb.set_trace()
    
    # path2_points = path + 'points/' + str(i) + '.pts'
    # path2_label = path + 'points_label/' + str(i) + '.seg'
    
    # np.savetxt(path2_points, selected_rows[:,:-1], fmt='%.6f')
    # np.savetxt(path2_label, selected_rows[:,-1:], fmt='%d')
    
    