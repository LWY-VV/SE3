# import sapien.core as sapien
import numpy as np
from PIL import Image, ImageColor
# from sapien.utils.viewer import Viewer
from transforms3d.euler import mat2euler, euler2mat
import torch
import trimesh
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
    DifferentiableTwoLinkRobot,
)
# from pytorch3d.transforms import quaternion_to_matrix

import numpy as np
# from SDF import compute_unit_sphere_transform
import matplotlib.pyplot as plt
import pdb
import struct
import math
# from pytorch3d.transforms import matrix_to_axis_angle
# import pytorch3d.transforms
import torch.nn.functional as F


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def get_EE_pos(pre_rotmatrix):
    """
    calculate the position of the end-effector
    
    input: output of the nueral network 
    
    output: the coordinate of position of joint:panda_hand 
    """
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
    hand_pos = torch.tensor([final[0,3] ,final[1,3], final[2,3]]).cuda()
    return hand_pos


def EE_position(joints):
    """
    calculate the position of the end-effector
    
    input: [1,7] tensor, the joints angles of panda arm
    
    output: the coordinate of position of joint:panda_hand 
    
    this function used differential robot model 
    """
    mesh_path_panda = './panda/panda.urdf'
    panda_robot = DifferentiableRobotModel(urdf_path=mesh_path_panda, name="panda", device='cuda')
    panda_links_all = panda_robot.compute_forward_kinematics_all_links(q=joints)
    EE_pos = panda_links_all['panda_hand'][0]
    return EE_pos

# def find_angle(cos_a, neg_sin_a):
#     # Calculate the basic angle using arccos
#     basic_angle = torch.arccos(cos_a)

#     # Deduce the quadrant to get the final angle
#     if cos_a > 0 and neg_sin_a < 0:         # First quadrant
#         return basic_angle
#     elif cos_a < 0 and neg_sin_a < 0:       # Second quadrant
#         return math.pi - basic_angle
#     elif cos_a < 0 and neg_sin_a > 0:       # Third quadrant
#         return math.pi + basic_angle
#     elif cos_a > 0 and neg_sin_a > 0:       # Fourth quadrant
#         return 2*math.pi - basic_angle


# def get_theta(local_rotmatrix):
#     T_DH_nontheta = torch.tensor(np.load('rotmatrix_notheta.npy')).cuda()
#     rotate_only_theta = torch.divide(local_rotmatrix,T_DH_nontheta)
#     theta_all = []
#     for i in range(rotate_only_theta.shape[0]):
#         theta1 = torch.arccos(rotate_only_theta[i,0,0])
#         theta2 = -torch.arcsin(rotate_only_theta[i,0,1])

#         theta = (theta1 + theta2)/2

#         theta_all.append(theta)
#         print(theta1)
#         print(theta2)

#     return theta_all


# def get_theta(local_rotmatrix):
#     """
#     """
#     theta_all = []
#     for i in range(local_rotmatrix.shape[0]):
#         matrix = local_rotmatrix[i]
#         theta = torch.atan2(matrix[1,0], matrix[0,0])
#         theta_all.append(theta)
#     theta_all = torch.vstack(theta_all)
#     return theta_all

def get_local_rotamatrix(pre_rotmatrix):
    """
    calculate the local rotation matrix from parent link to child link
    
    input: the output of the nueral network
    
    output: dimension:[21,3] (need to reshape to 7*3*3 after use)
            
            
    
    """
    pre_rotmatrix = pre_rotmatrix[0]
    rot_local_all = []
    for i in range(pre_rotmatrix.shape[0]-1):
        rot_local = pre_rotmatrix[i].T @ pre_rotmatrix[i+1]
        rot_local_all.append(rot_local)
    rot_local_all = torch.vstack(rot_local_all)
    return rot_local_all

def transformation_matrix(R, d):
    x = torch.hstack((R, d.reshape(-1, 1)))
    return torch.vstack((x, torch.tensor([0., 0., 0., 1.]).cuda()))

def panda_forward_kinematic_withlocalmatrix(rot_local_all):
    
    """
    hand write the forward kinematic of panda arm:
    
    input: local rotation matrix
    
    output: pointcloud of all panda link
            
    """
#     pre_rotmatrix = pre_rotmatrix[0]
    sample_point_num = 500
#     rot_local_all = []
#     rot_local_all.append(pre_rotmatrix[0])
    
#     for i in range(pre_rotmatrix.shape[0]-1):
#         rot_local = pre_rotmatrix[i].T @ pre_rotmatrix[i+1]
#         rot_local_all.append(rot_local)

    offsets = torch.tensor([[0., 0., 0.333],
                        [0.,   0.,  0.],
                        [0, -0.316, 0.],
                        [0.0825,0., 0.],
                        [-0.0825, 0.384, 0],
                        [0., 0., 0.],
                        [0.088, 0., 0.],
                        [0.,0.,0.107]]).float().cuda()

    mesh_dic = { "path2_mesh0" : './panda/meshes/visual_stl/link0.stl',
      "path2_mesh1": './panda/meshes/visual_stl/link1.stl',
      "path2_mesh2" : './panda/meshes/visual_stl/link2.stl',
      "path2_mesh3" : './panda/meshes/visual_stl/link3.stl',
      "path2_mesh4" : './panda/meshes/visual_stl/link4.stl',
      "path2_mesh5" : './panda/meshes/visual_stl/link5.stl',
      "path2_mesh6" : './panda/meshes/visual_stl/link6.stl',
      "path2_mesh7" : './panda/meshes/visual_stl/link7.stl',
      "path2_hand" : './panda/meshes/visual_stl/hand.stl' }


    pc_last2 = []
    mesh_link7 = trimesh.load(mesh_dic["path2_mesh7"], force='mesh')
    pc_link7 = torch.tensor(trimesh.sample.sample_surface(mesh_link7,sample_point_num)[0]).float().cuda()
    pc_last2.append(pc_link7)
    mesh_hand = trimesh.load(mesh_dic["path2_hand"], force='mesh')
    pc_hand = torch.tensor(trimesh.sample.sample_surface(mesh_hand,sample_point_num)[0]).float().cuda()
    rotate_hand = torch.tensor(euler2mat(0, 0, -0.785398163397)).float().cuda()
    pc_last2.append((rotate_hand@ (pc_hand + torch.tensor([0., 0., 0.107]).float().cuda()).T).T)
    pc_last2 = torch.vstack(pc_last2)

    Translation_all = []
    for i in range(len(offsets)-1):

        m = transformation_matrix(rot_local_all[i+1],offsets[i])
        Translation_all.append(m)

    trans_hand = transformation_matrix(rot_local_all[-1],offsets[-1])
    Translation_all.append(trans_hand)
#     pdb.set_trace()
#     Translation_all = torch.tensor(Translation_all)

    pc_all = []
    for name in mesh_dic:
        mesh = trimesh.load(mesh_dic[name], force='mesh')
        
        pc = torch.tensor(trimesh.sample.sample_surface(mesh,sample_point_num)[0]).float().cuda()
        pc_all.append(pc)

#     pc_all = torch.tensor(pc_all).float().cuda()

    trans = torch.eye(4)

    pc_world_all = []
    pc_world_all.append(pc_all[0])

    pc_1 = Translation_all[0] @ (torch.hstack((pc_all[1], torch.ones((sample_point_num,1)).cuda()))).T
    pc_world_all.append((pc_1.T)[:,:-1])


    pc_2 = Translation_all[0]@ Translation_all[1] @ (torch.hstack((pc_all[2], torch.ones((sample_point_num,1)).cuda()))).T
    pc_world_all.append((pc_2.T)[:,:-1])

    pc_3 = Translation_all[0]@ Translation_all[1]@ Translation_all[2] @ (torch.hstack((pc_all[3], torch.ones((sample_point_num,1)).cuda()))).T
    pc_world_all.append((pc_3.T)[:,:-1])

    pc_4 = Translation_all[0]@ Translation_all[1]@ Translation_all[2] @ Translation_all[3] @ (torch.hstack((pc_all[4], torch.ones((sample_point_num,1)).cuda()))).T
    pc_world_all.append((pc_4.T)[:,:-1])


    pc_5 = Translation_all[0]@ Translation_all[1]@ Translation_all[2] @ Translation_all[3] @ Translation_all[4] @ (torch.hstack((pc_all[5], torch.ones((sample_point_num,1)).cuda()))).T
    pc_world_all.append((pc_5.T)[:,:-1])

    pc_6 = Translation_all[0]@ Translation_all[1]@ Translation_all[2] @ Translation_all[3] @ Translation_all[4]@ Translation_all[5] @ (torch.hstack((pc_all[6], torch.ones((sample_point_num,1)).cuda()))).T
    pc_world_all.append((pc_6.T)[:,:-1])

    pc_7 = Translation_all[0]@ Translation_all[1]@ Translation_all[2] @ Translation_all[3] @ Translation_all[4]@ Translation_all[5] @ Translation_all[6] @ (torch.hstack((pc_last2,torch.ones((sample_point_num*2,1)).cuda()))).T
    pc_world_all.append((pc_7.T)[:,:-1])

    pc_world_all = torch.vstack(pc_world_all)
    return pc_world_all

def panda_forward_kinematic(pre_rotmatrix):
    """
    hand write the forward kinematic of panda arm:
    
    input: the output of the nueral network 
    
    output: pointcloud of all panda link
            
    """
    pre_rotmatrix = pre_rotmatrix[0]
    sample_point_num = 500
    rot_local_all = []
    rot_local_all.append(pre_rotmatrix[0])
    
    for i in range(pre_rotmatrix.shape[0]-1):
        rot_local = pre_rotmatrix[i].T @ pre_rotmatrix[i+1]
        rot_local_all.append(rot_local)

    offsets = torch.tensor([[0., 0., 0.333],
                        [0.,   0.,  0.],
                        [0, -0.316, 0.],
                        [0.0825,0., 0.],
                        [-0.0825, 0.384, 0],
                        [0., 0., 0.],
                        [0.088, 0., 0.],
                        [0.,0.,0.107]]).float().cuda()

    mesh_dic = { "path2_mesh0" : './panda/meshes/visual_stl/link0.stl',
      "path2_mesh1": './panda/meshes/visual_stl/link1.stl',
      "path2_mesh2" : './panda/meshes/visual_stl/link2.stl',
      "path2_mesh3" : './panda/meshes/visual_stl/link3.stl',
      "path2_mesh4" : './panda/meshes/visual_stl/link4.stl',
      "path2_mesh5" : './panda/meshes/visual_stl/link5.stl',
      "path2_mesh6" : './panda/meshes/visual_stl/link6.stl',
      "path2_mesh7" : './panda/meshes/visual_stl/link7.stl',
      "path2_hand" : './panda/meshes/visual_stl/hand.stl' }


    pc_last2 = []
    mesh_link7 = trimesh.load(mesh_dic["path2_mesh7"], force='mesh')
    pc_link7 = torch.tensor(trimesh.sample.sample_surface(mesh_link7,sample_point_num)[0]).float().cuda()
    pc_last2.append(pc_link7)
    mesh_hand = trimesh.load(mesh_dic["path2_hand"], force='mesh')
    pc_hand = torch.tensor(trimesh.sample.sample_surface(mesh_hand,sample_point_num)[0]).float().cuda()
    rotate_hand = torch.tensor(euler2mat(0, 0, -0.785398163397)).float().cuda()
    pc_last2.append((rotate_hand@ (pc_hand + torch.tensor([0., 0., 0.107]).float().cuda()).T).T)
    pc_last2 = torch.vstack(pc_last2)

    Translation_all = []
    for i in range(len(offsets)-1):

        m = transformation_matrix(rot_local_all[i+1],offsets[i])
        Translation_all.append(m)

    trans_hand = transformation_matrix(rot_local_all[-1],offsets[-1])
    Translation_all.append(trans_hand)
#     pdb.set_trace()
#     Translation_all = torch.tensor(Translation_all)

    pc_all = []
    for name in mesh_dic:
        mesh = trimesh.load(mesh_dic[name], force='mesh')
        
        pc = torch.tensor(trimesh.sample.sample_surface(mesh,sample_point_num)[0]).float().cuda()
        pc_all.append(pc)

#     pc_all = torch.tensor(pc_all).float().cuda()

    trans = torch.eye(4)

    pc_world_all = []
    pc_world_all.append(pc_all[0])

    pc_1 = Translation_all[0] @ (torch.hstack((pc_all[1], torch.ones((sample_point_num,1)).cuda()))).T
    pc_world_all.append((pc_1.T)[:,:-1])


    pc_2 = Translation_all[0]@ Translation_all[1] @ (torch.hstack((pc_all[2], torch.ones((sample_point_num,1)).cuda()))).T
    pc_world_all.append((pc_2.T)[:,:-1])

    pc_3 = Translation_all[0]@ Translation_all[1]@ Translation_all[2] @ (torch.hstack((pc_all[3], torch.ones((sample_point_num,1)).cuda()))).T
    pc_world_all.append((pc_3.T)[:,:-1])

    pc_4 = Translation_all[0]@ Translation_all[1]@ Translation_all[2] @ Translation_all[3] @ (torch.hstack((pc_all[4], torch.ones((sample_point_num,1)).cuda()))).T
    pc_world_all.append((pc_4.T)[:,:-1])


    pc_5 = Translation_all[0]@ Translation_all[1]@ Translation_all[2] @ Translation_all[3] @ Translation_all[4] @ (torch.hstack((pc_all[5], torch.ones((sample_point_num,1)).cuda()))).T
    pc_world_all.append((pc_5.T)[:,:-1])

    pc_6 = Translation_all[0]@ Translation_all[1]@ Translation_all[2] @ Translation_all[3] @ Translation_all[4]@ Translation_all[5] @ (torch.hstack((pc_all[6], torch.ones((sample_point_num,1)).cuda()))).T
    pc_world_all.append((pc_6.T)[:,:-1])

    pc_7 = Translation_all[0]@ Translation_all[1]@ Translation_all[2] @ Translation_all[3] @ Translation_all[4]@ Translation_all[5] @ Translation_all[6] @ (torch.hstack((pc_last2,torch.ones((sample_point_num*2,1)).cuda()))).T
    pc_world_all.append((pc_7.T)[:,:-1])

    # pc_hand = Translation_all[0]@ Translation_all[1]@ Translation_all[2] @ Translation_all[3] @ Translation_all[4]@ Translation_all[5] @ Translation_all[6]@ Translation_all[7]  @ (np.hstack((pc_all[8], np.ones((500,1))))).T
    # pc_world_all.append((pc_hand.T)[:,:-1])

    # pc_3 = Translation_all[0]@ Translation_all[1]@ Translation_all[2] @ (np.hstack((pc_all[3], np.ones((500,1))))).T
    # pc_world_all.append((pc_3.T)[:,:-1])


    # for i in range(7):
    #     for j in Translation_all[:i,:,:]:
    #         trans = j @ trans
    #     points_4d = np.hstack((pc_all[i], np.ones((500,1))))
    #     pc_world = (trans @ points_4d.T).T[:,:-1]

    #     pc_world_all.append(pc_world)
    pc_world_all = torch.vstack(pc_world_all)
    return pc_world_all

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))




def door_pc_get(file_path, pose):
    """
    
    """
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    # urdf_path = '/home/wangyi/se3_equivariant_place_recognition-master/data_generate_IK/door/8994/frame.urdf'
    # load as a kinematic articulation
    asset = loader.load_kinematic(file_path)
    assert asset, 'URDF not loaded.'
    # pose = [0.7,0.]
    asset.set_qpos(pose)


    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    # ---------------------------------------------------------------------------- #
    # Camera
    # ---------------------------------------------------------------------------- #
    near, far = 0.1, 100
    width, height = 640, 480
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )
    camera.set_pose(sapien.Pose(p=[1, 0, 0]))

    print('Intrinsic matrix\n', camera.get_intrinsic_matrix())

    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    camera.set_parent(parent=camera_mount_actor, keep_pose=False)

    # Compute the camera pose by specifying forward(x), left(y) and up(z)
    cam_pos = np.array([-2, -2, 3])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))

    scene.step()  # make everything set
    scene.update_render()
    camera.take_picture()

    # ---------------------------------------------------------------------------- #
    # RGBA
    # ---------------------------------------------------------------------------- #
    rgba = camera.get_float_texture('Color')  # [H, W, 4]
    # An alias is also provided
    # rgba = camera.get_color_rgba()  # [H, W, 4]
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    rgba_pil = Image.fromarray(rgba_img)
    # rgba_pil.save('color.png')

    # ---------------------------------------------------------------------------- #
    # XYZ position in the camera space
    # ---------------------------------------------------------------------------- #
    # Each pixel is (x, y, z, render_depth) in camera space (OpenGL/Blender)
    position = camera.get_float_texture('Position')  # [H, W, 4]

    # OpenGL/Blender: y up and -z forward
    points_opengl = position[..., :3][position[..., 3] < 1]
    points_color = rgba[position[..., 3] < 1][..., :3]
    # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
    # camera.get_model_matrix() must be called after scene.update_render()!
    model_matrix = camera.get_model_matrix()
    points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    
    # from PC_generate import pc_visualize
    # pc_visualize(points_world)

    # SAPIEN CAMERA: z up and x forward
    # points_camera = points_opengl[..., [2, 0, 1]] * [-1, -1, 1]
    return points_world[::3]

def pc_visualize(points):
    """
    visualize the pointcloud 
    """
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
    
    """
    get the pointcloud of panda arm with input of joint angles
    """
    mesh_path_panda = './panda/panda.urdf'
    panda_robot = DifferentiableRobotModel(urdf_path=mesh_path_panda, name="panda", device='cuda')
    panda_q = torch.tensor(joint_all).cuda()
    panda_links_all = panda_robot.compute_forward_kinematics_all_links(q=panda_q)
    EE_pos = panda_links_all['panda_hand'][0]
    points_all = []
    link_idx = 1
    for link_name in panda_links_all:
        if link_name == 'world'or link_name == 'panda_link8' or link_name == 'panda_leftfinger'or link_name == 'panda_rightfinger':
                continue      
        new_string = link_name.replace("panda_", "")
        path2_mesh = './panda/meshes/visual_stl/' + new_string +'.stl'
        
        mesh = trimesh.load(path2_mesh, force='mesh')
        pc = trimesh.sample.sample_surface(mesh,500)[0]


        orientation = panda_links_all[link_name][1][0]
        link_orientation = torch.cat((orientation[-1:], orientation[:-1]))
        rotat_matrix = quaternion_to_matrix(link_orientation).cpu().tolist()
        link_position = panda_links_all[link_name][0][0].cpu().tolist()
        # rot_matrix_2 = np.eye(3).cpu()
        point_world = (rotat_matrix @ pc.T).T + link_position
        
        idxs = np.ones((pc.shape[0], 1))*link_idx
        pc_with_idx = np.concatenate((point_world, idxs), axis=1)
        
        points_all.append(pc_with_idx)
        link_idx += 1
        if link_idx == 9:
            link_idx = 8

    points_all = np.vstack(points_all)
    return points_all, EE_pos


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

def rotatrix2jointstate(rotmatrix):
    joints_all = []
    for i in range(rotmatrix.shape[0]):
        joint = pytorch3d.transforms.matrix_to_axis_angle(rotmatrix[i]).detach()
        joints_all.append(joint)
    joints_all = np.array(joints_all)
    return joints_all


def Chordal_distance(R1, R2):
    
    """
    calculate the Chordal_distance of two matrix
    """
    B = R1.shape[0]
    part_num = R1.shape[1]
    err = 0
    for b in range(B):
        for p in range(part_num):
            diff = R1[b,p,:,:] - R2[b,p,:,:]
            distance = torch.sqrt(torch.trace(diff.T @ diff))
            err += distance
    
    return err

def test_pc_generate(pre_rotmatrx, joint_gt):
        
    pre_rotmatrx = pre_rotmatrx[0]
    mesh_path_panda = './panda/panda.urdf'
    panda_robot = DifferentiableRobotModel(urdf_path=mesh_path_panda, name="panda", device='cuda')
    panda_q = torch.tensor(joint_gt).cuda()
    panda_links_all = panda_robot.compute_forward_kinematics_all_links(q=panda_q)
    points_all = []
    link_idx = 0
    for link_name in panda_links_all:
        if link_name == 'world'or link_name == 'panda_link8' or link_name == 'panda_leftfinger'or link_name == 'panda_rightfinger':
                continue      
        new_string = link_name.replace("panda_", "")
        path2_mesh = './panda/meshes/visual_stl/' + new_string +'.stl'
        
        mesh = trimesh.load(path2_mesh, force='mesh')
        pc = torch.tensor(trimesh.sample.sample_surface(mesh,500)[0]).float().cuda()


        orientation = pre_rotmatrx[link_idx]

        ########## debug
#         pred_euler = matrix_to_euler_angles(pre_rotmatrx[link_idx].cpu().detach(), 'XYZ')
#         gt_euler = matrix_to_euler_angles(RT_of[0,link_idx].cpu(), 'XYZ')
#         print(pre_rotmatrx[link_idx].cpu().detach().reshape(1,-1))
#         print(RT_of[0,link_idx].cpu().reshape(1,-1))
#         print("-- ", (pred_euler-gt_euler)*180/np.pi, '\n')
        
        
        
        # link_orientation = torch.cat((orientation[-1:], orientation[:-1]))
        # rotat_matrix = quaternion_to_matrix(link_orientation).cpu().tolist()
        link_position = panda_links_all[link_name][0][0]
        # pdb.set_trace()
        point_world = (orientation @ pc.T).T + link_position
        
        # idxs = np.ones((pc.shape[0], 1))*link_idx
        # pc_with_idx = np.concatenate((point_world, idxs), axis=1)
        
        points_all.append(point_world)
        link_idx += 1
        if link_idx == 8:
            break
        

    points_all = torch.vstack(points_all)
    return points_all