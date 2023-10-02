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

