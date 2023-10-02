"""Camera.

Concepts:
    - Create and mount cameras
    - Render RGB images, point clouds, segmentation masks
"""

import sapien.core as sapien
import numpy as np
from PIL import Image, ImageColor
import open3d as o3d
from sapien.utils.viewer import Viewer
from transforms3d.euler import mat2euler


def door_pc_get(file_path, pose):
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

    