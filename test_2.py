import torch
import pytorch3d
import trimesh
import pybullet as p
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
    DifferentiableTwoLinkRobot,
    DifferentiableFrankaPanda
)
from pytorch3d.transforms import quaternion_to_matrix
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
from util import pandaPC_generate, pc_visualize
from SE3_model import EPNGeM
import time

def reset_pos_ori(robot,pos, ori):
    new_orientation_quat = p.getQuaternionFromEuler(ori)
    p.resetBasePositionAndOrientation(robot, pos, new_orientation_quat)




# panda_robot = DifferentiableFrankaPanda(device='cuda')
# panda_q = torch.tensor(joint_overfit).cuda()
# panda_links_all = panda_robot.compute_forward_kinematics_all_links(q=panda_q)

# panda_links_all_2 = panda_robot.compute_forward_kinematics_joint_pose(q=panda_q, link_name="panda_link7", recursive=True)

panda_state = [ 0.02593538,  1.00374918,  1.14027857, -0.8,  0.35527771, 0.54639669,  0.83701636]
# panda_state = [ 0.,0.,0.,0.,0.,0.,0.]


mesh_path_panda = './panda/panda.urdf'
panda_robot1 = DifferentiableRobotModel(urdf_path=mesh_path_panda, name="panda", device='cuda')
panda_q = torch.tensor(panda_state).cuda()
panda_links_all = panda_robot1.compute_forward_kinematics_all_links(q=panda_q)


############ pybullet visualize
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

plane_id = p.loadURDF("plane.urdf")
panda_bullet = p.loadURDF("./panda/panda.urdf")

panda_state.insert(0,0)
for joint_index, angle in enumerate(panda_state):
    p.resetJointState(panda_bullet, joint_index, angle, 0.0)


state0 = p.getLinkState(panda_bullet,0)
state1 = p.getLinkState(panda_bullet,1)
state2 = p.getLinkState(panda_bullet,2)
state3 = p.getLinkState(panda_bullet,3)
state4 = p.getLinkState(panda_bullet,4)
state5 = p.getLinkState(panda_bullet,5)
state6 = p.getLinkState(panda_bullet,6)
state7 = p.getLinkState(panda_bullet,7)
state8 = p.getLinkState(panda_bullet,8)

# state0 = panda_links_all['panda_link0'][0]
# state1 = panda_links_all['panda_link1'][0]
# state2 = panda_links_all['panda_link2'][0]
# state3 = panda_links_all['panda_link3'][0]
# state4 = panda_links_all['panda_link4'][0]
# state5 = panda_links_all['panda_link5'][0]


# state0 = p.getJointInfo(panda_bullet,0)
# state1 = p.getJointInfo(panda_bullet,1)
# state2 = p.getJointInfo(panda_bullet,2)
# state3 = p.getJointInfo(panda_bullet,3)
# state4 = p.getJointInfo(panda_bullet,4)
# state5 = p.getJointInfo(panda_bullet,5)
# pdb.set_trace()
######################### create a ball to debug

p.setAdditionalSearchPath(pybullet_data.getDataPath())

radius = 0.08  # radius of the sphere
collisionShapeId = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)

# Create a multibody with the sphere collision shape
# ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=state0[0])
ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=state1[0])
# ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=state2[0])
# ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=state3[0])
# ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=state4[0])
# ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=state5[0])
# ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=state6[0])
# ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=state7[0])
# ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=state8[0])
# # ##########################

pdb.set_trace()
# time.sleep(10)
# reset_pos_ori(door,door_pos,door_ori)
