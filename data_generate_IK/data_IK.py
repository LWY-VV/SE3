import roboticstoolbox as rtb
import pdb
from spatialmath import SE3
import pybullet as p
import pybullet_data
import time
import numpy as np
from PC_generate import pandaPC_generate,pc_visualize
from door_PCget import door_pc_get
import matplotlib.pyplot as plt
from sympy import Plane, Point, Point3D
import transforms3d as t3d

def door_jointset(theta):
    p.resetJointState(door, 1, theta, 0.0)

def get_handel_pos(theta):
    door_state = [theta,0.]
    door_jointset(door_state[0])

    state_1 = p.getLinkState(door,0)
    state_2 = p.getLinkState(door,1)
    state_3 = p.getLinkState(door,2)

    # pdb.set_trace()

    handle_position = list(state_3[0])
    door_axis_position = list(state_2[0])
    
    #### projection
    pr = Point(handle_position[0], handle_position[1],handle_position[2])
    p1 = Plane(Point3D(door_axis_position[0], door_axis_position[1], door_axis_position[2]), normal_vector =(0, 0, 1))
    projectionPoint = p1.projection(pr)
    projection = np.array([float(projectionPoint[0]),float(projectionPoint[1]),float(projectionPoint[2])])
    
    vector_alongboard = projection - door_axis_position
    
    v1 = door_axis_position - projection
    v2 = handle_position - projection
    vector_orth2board = np.cross(v1, v2)
    
    vector_orth2board_norm = np.linalg.norm(vector_orth2board)
    vector_orth2board = vector_orth2board/vector_orth2board_norm
    catch_point = handle_position + vector_orth2board * 0.1
    
    return catch_point, vector_orth2board
    
def reset_pos_ori(robot,pos, ori):
    new_orientation_quat = p.getQuaternionFromEuler(ori)
    p.resetBasePositionAndOrientation(robot, pos, new_orientation_quat)

def rotomatrix_fromRpyPos(rpy):
    
    rot_matrix = t3d.euler.euler2mat(rpy[0],rpy[1],rpy[2])

    return rot_matrix


############ pybullet visualize
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

plane_id = p.loadURDF("plane.urdf")
panda_bullet = p.loadURDF("/home/wangyi/ICRA_basic_function/panda/panda.urdf")
door = p.loadURDF('/home/wangyi/se3_equivariant_place_recognition-master/data_generate_IK/door/8994/mobility.urdf')

door_pos =  [-0.5, -0.5, 0.93]
# door_ori = [0.2,0.3,0.3]
door_ori = [0.0,0.0,0.0]
reset_pos_ori(door,door_pos,door_ori)

########### door joints initial 


############ begin inverse kinematic
# Make a Panda robot
panda = rtb.models.Panda()
# Make a goal pose

def traj_get(start, end):
    handle_position_ini, direction_ini = get_handel_pos(start)
    Tep_ini = SE3.Trans(handle_position_ini[0], handle_position_ini[1], handle_position_ini[2]) * SE3.RPY([door_ori[0],door_ori[1],door_ori[2]+start])
    
    # Solve the IK problem of initial state
    panda_states_ini = panda.ikine_LM(Tep_ini).q
    parts = abs(int(end-start))//5
    panda_states = []
    theta_all = []
    for i in range(parts):
        if start < end:   
            theta = (start + 5*i)*np.pi/180
        else:
            theta = (start - 5*i)*np.pi/180
        theta_all.append(theta)
        handle_pos, direction = get_handel_pos(theta)
        Tep = SE3.Trans(handle_pos[0], handle_pos[1], handle_pos[2]) * SE3.RPY([door_ori[0],door_ori[1],door_ori[2]+theta])
        panda_state = panda.ikine_LM(Tep,q0 = panda_states_ini).q
        panda_states.append(panda_state)
        panda_states_ini = panda_state
        
        
        
    panda_states = np.array(panda_states)
    theta_all = np.array(theta_all)
    
    return theta_all,panda_states
        
door_tra,panda_tra = traj_get(0,-20)       
        

# qt = rtb.jtraj(panda.qr, q_pickup, 50)

# panda_states_bullet = panda_state.insert(0,0)
# for joint_index, angle in enumerate(panda_states_bullet):
#     # p.setJointMotorControl2(panda, joint_index, p.POSITION_CONTROL, targetPosition=angle)
#     p.resetJointState(panda_bullet, joint_index, angle, 0.0)


########## pointcloud generate
pc_all_all = []
door_rotomatrix = rotomatrix_fromRpyPos(door_ori)
for i in range(len(door_tra)):
    
    panda_pc_label = pandaPC_generate(panda_tra[i])
    panda_pc = panda_pc_label[:,:-1]
    frame_file = '/home/wangyi/se3_equivariant_place_recognition-master/data_generate_IK/door/8994/frame.urdf'
    board_file = '/home/wangyi/se3_equivariant_place_recognition-master/data_generate_IK/door/8994/board.urdf'
    frame_pc = door_pc_get(frame_file,[door_tra[i],0]) 
    board_pc = door_pc_get(board_file,[door_tra[i],0])
    door_pc = (door_rotomatrix @ np.concatenate((board_pc,frame_pc)).T).T  + door_pos

    pc_all = np.concatenate((door_pc,panda_pc))
    pc_all = panda_pc
    # pc_all = door_pc
    pc_all_all.append(pc_all)
pc_visualize(pc_all_all[-1],1)

# pc_all_all = np.array(pc_all_all)

# for pc in pc_all_all:
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     xs, ys, zs = pc[:, 0], pc[:, 1], pc[:, 2]
#     ax.set_xlim([-1, 1])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1, 1])
#     plt.show()
#     time.sleep(0.5)



######################### create a ball to debug

# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# radius = 0.05  # radius of the sphere
# collisionShapeId = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)

# # Create a multibody with the sphere collision shape
# ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=handle_position)
# ##########################
