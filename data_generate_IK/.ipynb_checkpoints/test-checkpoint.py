import roboticstoolbox as rtb
import pdb
from spatialmath import SE3
import pybullet as p
import pybullet_data
import time
import numpy as np
from PC_generate import pandaPC_generate,pc_visualize
from door_PCget import door_pc_get
from sympy import Plane, Point, Point3D



p.connect(p.GUI)

# Load the robot
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")
robot = p.loadURDF("/home/wangyi/ICRA_basic_function/panda/panda.urdf")

# Specify the new position (x, y, z) in meters
# new_position = [1, 2, 3]
new_position = [0, 0, 0]

# Specify the new orientation as Euler angles (roll, pitch, yaw) in radians
new_orientation_rpy = [0, 0, 0]

# Convert the Euler angles to a quaternion
new_orientation_quat = p.getQuaternionFromEuler(new_orientation_rpy)

# Apply the transformation
p.resetBasePositionAndOrientation(robot, new_position, new_orientation_quat)

time.sleep(30)


# def door_jointset(theta):
#     p.resetJointState(door, 1, theta, 0.0)
# door_import_offset =  [-0.4, -0.4, 0.93]
# ############ pybullet visualize
# physics_client = p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# plane_id = p.loadURDF("plane.urdf")
# # panda_bullet = p.loadURDF("/home/wangyi/ICRA_basic_function/panda/panda.urdf", [0, 0, 0], )
# door = p.loadURDF('/home/wangyi/se3_equivariant_place_recognition-master/data_generate_IK/door/8994/mobility.urdf',door_import_offset,)

# ########### door joints initial 

# door_state = [-0.5,0.]
# door_jointset(door_state[0])

# state_1 = p.getLinkState(door,0)
# state_2 = p.getLinkState(door,1)
# state_3 = p.getLinkState(door,2)


# handle_position = list(state_3[0])
# door_axis_position = list(state_2[0])


# ##### projection 
# pr = Point(handle_position[0], handle_position[1],handle_position[2])
  
# # using Plane()
# p1 = Plane(Point3D(door_axis_position[0], door_axis_position[1], door_axis_position[2]), normal_vector =(0, 0, 1))
  
# # using projection()

# projectionPoint = p1.projection(pr)
# projection = np.array([float(projectionPoint[0]),float(projectionPoint[1]),float(projectionPoint[2])])

# v1 = door_axis_position - projection
# v2 = handle_position - projection

# vector = np.cross(v1, v2)

# catch_point = handle_position + vector *1
# # import pdb; pdb.set_trace()

# ######################### create a ball to debug

# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# radius = 0.05  # radius of the sphere
# collisionShapeId = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)

# # Create a multibody with the sphere collision shape
# ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=handle_position)
# # ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=door_axis_position)

# # ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=projection)

# ball_handle = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, basePosition=catch_point)

# # ##########################
# time.sleep(60)