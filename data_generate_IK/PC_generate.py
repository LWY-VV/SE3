import torch
import trimesh
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
    DifferentiableTwoLinkRobot,
)
from pytorch3d.transforms import quaternion_to_matrix
import numpy as np
# from SDF import compute_unit_sphere_transform
import matplotlib.pyplot as plt
import pdb
import struct
begin_state_all = []
test_case_number = 500
for i in range(267,500):
    jointvalue1 = np.random.rand(1)*2.5 
    jointvalue2 = np.random.rand(1)*2 
    jointvalue3 = np.random.rand(1)*2.3
    jointvalue4 = np.random.rand(1)*-2.2
    jointvalue5 = np.random.rand(1)*2.5
    jointvalue6 = np.random.rand(1)*2.5
    jointvalue7 = np.random.rand(1)*2
    
    joint_all = np.array([jointvalue1,jointvalue2,jointvalue3,jointvalue4,jointvalue5,jointvalue6,jointvalue7]).reshape(1,7)
    # begin_state_all.append(joint_all)

# begin_state_all = np.array(begin_state_all).reshape(test_case_number,7)\
    
    
def pc_visualize(points,filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    # labels = points[:,3]
    jointposition_pybullrt = [-0.395648158558341, -0.7542892495494744, 1.753602974414826]
    offset = [-0.2542892495494744, -0.8236029744148256 ,0.10435184144165952]
    # handle = [-0.8908444056389443, -0.3777317738314502, 1.0269834999999998]

    
    origin= [0.,0.,0.]
    # x_ax = handle[0]
    # y_ax = handle[1]
    # z_ax = handle[2]
    
    ax.scatter(xs, ys, zs,c='b', marker='o',s=2)
    # ax.scatter(xs, ys, zs,c=zs, cmap='cividis', marker='*',s=10)
    
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



def pandaPC_generate(joint_all):
    
    mesh_path_panda = '/home/wangyi/se3_equivariant_place_recognition-master/panda/panda.urdf'
    panda_robot = DifferentiableRobotModel(urdf_path=mesh_path_panda, name="panda", device='cuda')
    panda_q = torch.tensor(joint_all).cuda()
    panda_links_all = panda_robot.compute_forward_kinematics_all_links(q=panda_q)
    points_all = []
    link_idx = 1
    for link_name in panda_links_all:
        if link_name == 'world'or link_name == 'panda_link8' or link_name == 'panda_leftfinger'or link_name == 'panda_rightfinger':
                continue      
        new_string = link_name.replace("panda_", "")
        path2_mesh = '/home/wangyi/se3_equivariant_place_recognition-master/panda/meshes/collision/' + new_string +'.stl'
        
        mesh = trimesh.load(path2_mesh, force='mesh')
        pc = trimesh.sample.sample_surface(mesh,500)[0]


        orientation = panda_links_all[link_name][1][0]
        link_orientation = torch.cat((orientation[-1:], orientation[:-1]))
        rotat_matrix = quaternion_to_matrix(link_orientation).cpu().tolist()
        link_position = panda_links_all[link_name][0][0].cpu().tolist()
        point_world = (rotat_matrix @ pc.T).T + link_position
        
        idxs = np.ones((pc.shape[0], 1))*link_idx
        pc_with_idx = np.concatenate((point_world, idxs), axis=1)
        
        points_all.append(pc_with_idx)
        link_idx += 1

    points_all = np.vstack(points_all)
    return points_all


def doorPC_generate(joint_all):
    
    mesh_path_door = '/home/wangyi/se3_equivariant_place_recognition-master/data_generate_IK/door/8867/mobility.urdf'
    door_robot = DifferentiableRobotModel(urdf_path=mesh_path_door, name="door", device='cuda')
    door_q = torch.tensor(joint_all).cuda()
    door_links_all = door_robot.compute_forward_kinematics_all_links(q=door_q)
    points_all = []
    link_idx = 1
    pc_coordinates = []
    pc_labels =[]

    with open("/home/wangyi/se3_equivariant_place_recognition-master/data_generate_IK/door/8994/point_sample/pts-10000.txt", "r") as f:
        for line in f:
            x, y, z = map(float, line.strip().split(" "))
            pc_coordinates.append([x, y, z])
    with open("/home/wangyi/se3_equivariant_place_recognition-master/data_generate_IK/door/8994/point_sample/label-10000.txt", "r") as f:
        for line in f:
            l = float(line.strip())
            pc_labels.append(l)
    pc_coordinates = np.array(pc_coordinates)
    pc_labels = np.array(pc_labels)
    door_base = pc_coordinates[np.where(pc_labels==1)]
    door_board = pc_coordinates[np.where(pc_labels==7)]

    # base_offset = np.array([0.34835063769460645, 0.824049, 0.01805896177769635])
    # door_base += base_offset
    for link_name in door_links_all:
        if link_name == 'base' or link_name == 'link_2':
                continue
        
        name_space = {
                        "link_0": door_base,
                        "link_1": door_board,
                        }
        
        pc = name_space[link_name]
        
        orientation = door_links_all[link_name][1][0]
        link_orientation = torch.cat((orientation[-1:], orientation[:-1]))
        rotat_matrix = quaternion_to_matrix(link_orientation).cpu().tolist()
        link_position = door_links_all[link_name][0][0].cpu().tolist()
        point_world = (rotat_matrix @ pc.T).T + link_position
        
        idxs = np.ones((pc.shape[0], 1))*link_idx
        pc_with_idx = np.concatenate((point_world, idxs), axis=1)
        
        points_all.append(pc_with_idx)
        link_idx += 1

    points_all = np.vstack(points_all)
    return points_all



