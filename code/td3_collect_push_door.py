import os
import numpy as np
from PIL import Image
from utils import save_h5, radian2degree, degree2radian
import json
from argparse import ArgumentParser
import torch
import time

from sapien.core import Pose, ArticulationJointType
from td3 import ReplayBuffer
from td3 import TD3
from env import Env, ContactError
from camera import Camera
from robots.panda_robot import Robot
import random
from scipy.spatial.transform import Rotation as R
import imageio
import multiprocessing as mp

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--primact_type', type=str)
parser.add_argument('--action_type', type=str, default='xxx')
parser.add_argument('--out_dir', type=str)
parser.add_argument('--trial_id', type=int, default=0, help='trial id')
parser.add_argument('--out_folder', type=str, default='xxx')
parser.add_argument('--dataset_id', type=int, default=0, help='dataset id')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='xxx')
parser.add_argument('--state_initial_position', action='store_true', default=False, help='xxx')
parser.add_argument('--state_initial_dir', action='store_true', default=False, help='xxx')
parser.add_argument('--state_initial_up_dir', action='store_true', default=False, help='xxx')
parser.add_argument('--state_joint_origins', action='store_true', default=False, help='xxx')
parser.add_argument('--state_ctpt_dis_to_joint', action='store_true', default=False, help='xxx')
parser.add_argument('--pred_world_xyz', type=int, default=0)
parser.add_argument('--pred_residual_world_xyz', type=int, default=0)
parser.add_argument('--pred_residual_root_qpos', type=int, default=1)
parser.add_argument('--up_norm_dir', action='store_true', default=False, help='xxx')
parser.add_argument('--final_dist', type=float, default=0.1)
parser.add_argument('--replay_buffer_size', type=int, default=5e5)
parser.add_argument('--pos_range', type=float, default=0.5)
parser.add_argument('--explore_noise_scale', type=float, default=0.01)
parser.add_argument('--eval_noise_scale', type=float, default=0.01)
parser.add_argument('--noise_decay', type=float, default=0.8)
# parser.add_argument('--sparse_reward', type=float, default=0.1)
parser.add_argument('--guidance_reward', type=float, default=0.0)
parser.add_argument('--decay_interval', type=int, default=100)
parser.add_argument('--q_lr', type=float, default=3e-4)
parser.add_argument('--policy_lr', type=float, default=3e-4)
parser.add_argument('--threshold', type=float, default=0.15)
parser.add_argument('--task_upper', type=float, default=30)
parser.add_argument('--task_lower', type=float, default=30)
parser.add_argument('--success_reward', type=int, default=10)
parser.add_argument('--target_margin', type=float, default=2)
parser.add_argument('--HER_move_margin', type=float, default=2)
parser.add_argument('--target_part_state', type=str, default='random-middle')
parser.add_argument('--num_steps', type=int, default=4)
parser.add_argument('--use_HER', action='store_true', default=False, help='xxx')
parser.add_argument('--HER_only_success', action='store_true', default=False, help='xxx')
parser.add_argument('--HER_only_attach', action='store_true', default=False, help='xxx')
parser.add_argument('--sample_num', type=int, default=2)
parser.add_argument('--wp_rot', action='store_true', default=False, help='xxx')
parser.add_argument('--use_direction_world', action='store_true', default=False, help='xxx')
parser.add_argument('--up_norm_thresh', type=float, default=0)
parser.add_argument('--use_random_up', action='store_true', default=False, help='xxx')
parser.add_argument('--all_shape', action='store_true', default=False, help='xxx')
parser.add_argument('--train_shape', action='store_true', default=False, help='xxx')
parser.add_argument('--val_shape', action='store_true', default=False, help='xxx')
parser.add_argument('--state_axes', action='store_true', default=False, help='xxx')
parser.add_argument('--state_axes_all', action='store_true', default=False, help='xxx')
parser.add_argument('--open_door', action='store_true', default=False, help='xxx')
parser.add_argument('--degree_uplimit', type=int, default=200)
parser.add_argument('--closeDoor_uplimit', type=int, default=1000)

parser.add_argument('--num_processes', type=int, default=1)

parser.add_argument('--vis_gif', action='store_true', default=False, help='xxx')
parser.add_argument('--curiosity', action='store_true', default=False, help='xxx')
parser.add_argument('--novel_cat', action='store_true', default=False, help='xxx')
parser.add_argument('--saved_epoch', type=str, default=10)
parser.add_argument('--rot_degree', type=float, default=30.0)


args = parser.parse_args()

def check_shape_valid(shape_id):
    exclusive_cabinet_shape_id_list = ['41452', '45166', '45323', '45503', '45642', '45690', '45855', '45910', '45915', '45922', '46019', '46044', '46092', '46107', '46108', '46127', '46132', '46166', '46180', '46403', '46408', '46417', '46430', '46439', '46481', '46699', '46768', '46922', '46944', '47185', '47254', '47388', '47391', '47419', '47585', '47601', '47632', '47686', '47729', '47954', '47963', '48036', '48063', '48167', '48479', '48686', '48700', '48878']
    exclusive_microwave_shape_id_list = ['7306', '7273', '7292']
    exclusive_table_shape_id_list_revolute = ['19384', '19740', '20077', '21718', '21473', '22301', '22433', '22870', '23650', '24152', '25160', '25756', '25959', '26545', '26800', '26886', '27189', '28594']
    if shape_id in exclusive_cabinet_shape_id_list:
        return False
    if shape_id in exclusive_microwave_shape_id_list:
        return False
    if shape_id in exclusive_table_shape_id_list_revolute:
        return False
    return True

def run_an_RL(idx_process, args, transition_Q):


    microwave_id_link_dict = {
        "7119": "link_0",
        "7128": "link_0",
        "7167": "link_0",
        "7221": "link_0",
        "7236": "link_0",
        "7263": "link_0",
        "7265": "link_0",
        "7273": "link_0",
        "7292": "link_0",
        "7296": "link_1",
        "7304": "link_0",
        "7306": "link_0",
        "7310": "link_0",
        "7320": "link_0",
        "7349": "link_1",
        "7366": "link_1",
    }

    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"

    np.random.seed(random.randint(1, 1000) + idx_process)
    random.seed(random.randint(1, 1000) + idx_process)

    train_shape_list, val_shape_list = [], []
    train_file_dir = "../stats/train_VAT_train_data_list.txt"
    val_file_dir = "../stats/train_VAT_test_data_list.txt"
    all_shape_list = []

    if ('Drawer' in args.action_type) or ('drawer' in args.action_type):
        all_cat_list = ['StorageFurniture']
        if args.novel_cat:
            all_cat_list = ['Table']
    if ('Door' in args.action_type) or ('door' in args.action_type):
        all_cat_list = ['StorageFurniture', 'Microwave', 'Refrigerator', 'Door']
        if args.novel_cat:
            all_cat_list = ['Safe', 'WashingMachine', 'Table']

    len_shape = {}
    len_train_shape = {}
    len_val_shape = {}
    shape_cat_dict = {}
    cat_shape_id_list = {}
    val_cat_shape_id_list = {}
    train_cat_shape_id_list = {}
    for cat in all_cat_list:
        len_shape[cat] = 0
        len_train_shape[cat] = 0
        len_val_shape[cat] = 0
        cat_shape_id_list[cat] = []
        train_cat_shape_id_list[cat] = []
        val_cat_shape_id_list[cat] = []

    with open(train_file_dir, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            if cat not in all_cat_list:
                continue
            train_shape_list.append(shape_id)
            all_shape_list.append(shape_id)
            shape_cat_dict[shape_id] = cat
            len_shape[cat] += 1
            len_train_shape[cat] += 1
            cat_shape_id_list[cat].append(shape_id)
            train_cat_shape_id_list[cat].append(shape_id)

    with open(val_file_dir, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            if cat not in all_cat_list:
                continue
            val_shape_list.append(shape_id)
            all_shape_list.append(shape_id)
            shape_cat_dict[shape_id] = cat
            len_shape[cat] += 1
            len_val_shape[cat] += 1
            cat_shape_id_list[cat].append(shape_id)
            val_cat_shape_id_list[cat].append(shape_id)

    EP_MAX = 1000000
    hidden_dim = 512
    DETERMINISTIC = True  # DDPG: deterministic policy gradient
    threshold_gripper_distance = args.threshold

    # task = np.pi * args.task / 180
    pos_range = args.pos_range
    rot_range = np.pi * args.rot_degree / 180
    action_range = torch.tensor([pos_range, pos_range, pos_range, rot_range, rot_range, rot_range]).to(device)
    action_dim = 6
    state_dim = 1 + 1 + 1 + 3 + 3 + 8  # cur_obj_qpos, dis_to_target, cur_gripper_info, cur_step_idx, final_task(degree), contact_point_xyz, gripper_xyz
    if args.state_initial_position:
        state_dim += 3
    if args.state_initial_dir:
        state_dim += 9
    if args.state_initial_up_dir:
        state_dim += 3
    if args.state_joint_origins:
        state_dim += 3
    if args.state_ctpt_dis_to_joint:
        state_dim += 1
    if args.state_axes:
        state_dim += 1
    if args.state_axes_all:
        state_dim += 3
    # state
    # 0     : door's current qpos
    # 1     : distance to task
    # 2     : task
    # 3-5   : start_gripper_root_position
    # 6-8   : contact_point_position_world
    # 9-11  : gripper_finger_position
    # 12-19 : gripper_qpos
    # 20-28 : up, forward, left
    #       : up
    # 29-31 : joint_origins
    # 32    : state_ctpt_dis_to_joint
    # 33-37 : step_idx
    trial_id = args.trial_id
    primact_type = args.primact_type
    out_dir = os.path.join(args.out_dir,
                           'RL_%s_%d' % (args.action_type, trial_id))

    replay_buffer_size = args.replay_buffer_size
    replay_buffer = ReplayBuffer(replay_buffer_size)
    td3 = TD3(replay_buffer, state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, action_range=action_range, device=device, pred_world_xyz=args.pred_world_xyz, wp_rot=args.wp_rot).to(device)
    td3.load_model(path=os.path.join(out_dir, 'td3_%s' % args.saved_epoch))

    # setup env
    print("creating env")
    env = Env(show_gui=(not args.no_gui))
    print("env creared")

    ### viz the EE gripper position
    # setup robot
    robot_urdf_fn = './robots/panda_gripper.urdf'
    robot_material = env.get_material(4, 4, 0.01)

    if args.pred_world_xyz + args.pred_residual_world_xyz + args.pred_residual_root_qpos != 1:
        raise ValueError

    robot_loaded = 0
    tot_done_epoch = 0
    tot_fail_epoch = 0
    object_material = env.get_material(4, 4, 0.01)
    cam = None

    prev_found = 0
    prev_cat = all_cat_list[0]

    for epoch in range(EP_MAX):

        torch.cuda.empty_cache()

        if cam is None:
            cam = Camera(env, random_position=True, restrict_dir=True)
        else:
            cam.change_pose(random_position=True, restrict_dir=True)
        if not args.no_gui:
            env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)

        torch.cuda.empty_cache()

        flog = open(os.path.join(out_dir, '%s_log_collect.txt' % str(idx_process)), 'a')
        env.flog = flog
        if prev_found == 0:
            selected_cat = prev_cat
        else:
            selected_cat = all_cat_list[random.randint(0, len(all_cat_list) - 1)]
        prev_cat = selected_cat
        prev_found = 0

        if args.train_shape:
            shape_id = train_cat_shape_id_list[selected_cat][random.randint(0, len_train_shape[selected_cat] - 1)]
            while check_shape_valid(shape_id) is False:
                shape_id = train_cat_shape_id_list[selected_cat][random.randint(0, len_train_shape[selected_cat] - 1)]
        elif args.val_shape:
            shape_id = val_cat_shape_id_list[selected_cat][random.randint(0, len_val_shape[selected_cat] - 1)]
            while check_shape_valid(shape_id) is False:
                shape_id = val_cat_shape_id_list[selected_cat][random.randint(0, len_val_shape[selected_cat] - 1)]
        elif args.all_shape:
            shape_id = cat_shape_id_list[selected_cat][random.randint(0, len_shape[selected_cat] - 1)]
            while check_shape_valid(shape_id) is False:
                shape_id = cat_shape_id_list[selected_cat][random.randint(0, len_shape[selected_cat] - 1)]
        else:
            print("Error")

        object_urdf_fn = '../data/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % shape_id
        target_part_state = args.target_part_state
        joint_angles = env.load_object(object_urdf_fn, object_material, state=target_part_state,
                                                              target_part_id=-1)
        env.render()

        still_timesteps = 0
        wait_timesteps = 0
        cur_qpos = env.get_object_qpos()
        while still_timesteps < 500 and wait_timesteps < 3000:
            env.step()
            env.render()
            cur_new_qpos = env.get_object_qpos()
            invalid_contact = False
            for c in env.scene.get_contacts():
                for p in c.points:
                    if abs(p.impulse @ p.impulse) > 1e-4:
                        invalid_contact = True
                        break
                if invalid_contact:
                    break
            if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and (not invalid_contact):
                still_timesteps += 1
            else:
                still_timesteps = 0
            cur_qpos = cur_new_qpos
            wait_timesteps += 1

        if still_timesteps < 500:
            print('Object Not Still!\n')
            env.scene.remove_articulation(env.object)
            flog.close()
            continue

        ### use the GT vision
        rgb, depth = cam.get_observation()
        # cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
        # cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])

        # get movable link mask
        object_link_ids = env.movable_link_ids
        gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)

        # sample a pixel to interact
        xs, ys = np.where(gt_movable_link_mask > 0)
        if len(xs) == 0:
            env.scene.remove_articulation(env.object)
            flog.close()
            continue
        prev_found = 1

        idx = np.random.randint(len(xs))
        x, y = xs[idx], ys[idx]
        target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
        env.set_target_object_part_actor_id(target_part_id)
        tot_trial = 0
        joint_type = env.target_object_part_joint_type
        link_name = env.target_object_part_actor_link.name
        joint_axes = env.get_target_part_axes_new(target_part_id=target_part_id)
        while (tot_trial < 50) and ((joint_type != ArticulationJointType.REVOLUTE) or ((selected_cat == "Microwave") and (link_name != microwave_id_link_dict[shape_id])) or ((selected_cat == "Door") and (abs(joint_axes[-1]) < 0.9))):
            idx = np.random.randint(len(xs))
            x, y = xs[idx], ys[idx]
            target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
            env.set_target_object_part_actor_id(target_part_id)
            tot_trial += 1
            joint_type = env.target_object_part_joint_type
            link_name = env.target_object_part_actor_link.name
            joint_axes = env.get_target_part_axes_new(target_part_id=target_part_id)
        if (tot_trial >= 50):
            env.scene.remove_articulation(env.object)
            flog.close()
            continue
        target_part_joint_idx = env.target_object_part_joint_id

        joint_angle_lower = env.joint_angles_lower[target_part_joint_idx]
        joint_angle_upper = env.joint_angles_upper[target_part_joint_idx]
        joint_angle_lower_degree = radian2degree(joint_angle_lower)
        joint_angle_upper_degree = radian2degree(joint_angle_upper - joint_angle_lower)
        task_upper = min(joint_angle_upper_degree, args.task_upper)

        task_lower = args.task_lower

        global degree_record
        task_degree = 0
        tot_trial = 0
        found_task = False
        while tot_trial < 50 and not found_task:
            task_degree = random.random() * (task_upper - task_lower) + task_lower
            tot_trial += 1
            if degree_record[int(task_degree // 10) - 1] < args.degree_uplimit:
                found_task = True
                break
        if not found_task:
            continue

        if args.open_door:
            joint_angles = env.update_joint_angle(joint_angles, target_part_joint_idx, target_part_state, task_degree, push=False, pull=True)
        else:
            joint_angles = env.update_joint_angle(joint_angles, target_part_joint_idx, target_part_state, task_degree, push=True, pull=False)
        # joint_angles = [0.9046]
        env.set_object_joint_angles(joint_angles)
        env.render()
        if args.open_door:
            task_degree = - task_degree
        task = degree2radian(task_degree)

        ### use the GT vision
        rgb, depth = cam.get_observation()
        cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
        cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
        gt_nor = cam.get_normal_map()

        # get movable link mask
        object_link_ids = env.movable_link_ids
        gt_movable_link_mask = cam.get_movable_link_mask([target_part_id])
        gt_all_movable_link_mask = cam.get_movable_link_mask(object_link_ids)

        # sample a pixel to interact
        xs, ys = np.where(gt_movable_link_mask > 0)
        if len(xs) == 0:
            print("len = 0")
            env.scene.remove_articulation(env.object)
            flog.close()
            continue

        env.render()
        idx = np.random.randint(len(xs))
        x, y = xs[idx], ys[idx]

        joint_origins = env.get_target_part_origins_new(target_part_id=target_part_id)
        joint_axes = env.get_target_part_axes_new(target_part_id=target_part_id)
        axes_dir = env.get_target_part_axes_dir_new(target_part_id)

        # get pixel 3D pulling direction (cam/world)
        direction_cam = gt_nor[x, y, :3]
        direction_cam /= np.linalg.norm(direction_cam)
        direction_world = cam.get_metadata()['mat44'][:3, :3] @ direction_cam

        # sample a random direction in the hemisphere (cam/world)
        action_direction_cam = np.random.randn(3).astype(np.float32)
        action_direction_cam /= np.linalg.norm(action_direction_cam)
        if action_direction_cam @ direction_cam > 0:
            action_direction_cam = -action_direction_cam
        action_direction_cam = -direction_cam
        action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam

        # get pixel 3D position (cam/world)
        position_cam = cam_XYZA[x, y, :3]
        position_cam_xyz1 = np.ones((4), dtype=np.float32)
        position_cam_xyz1[:3] = position_cam
        position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
        position_world = position_world_xyz1[:3]

        state_joint_origins = joint_origins
        state_joint_origins[-1] = position_world[-1]
        state_ctpt_dis_to_joint = np.linalg.norm(state_joint_origins - position_world)

        # compute final pose
        # here, RL give the action(qpos), then caculate the rotmat(SE3)
        up = np.array(action_direction_world, dtype=np.float32)  # up = action_direction_world
        forward = np.random.randn(3).astype(np.float32)
        while abs(up @ forward) > 0.99:
            forward = np.random.randn(3).astype(np.float32)
        left = np.cross(up, forward)
        left /= np.linalg.norm(left)  # get unit vector
        forward = np.cross(left, up)
        forward /= np.linalg.norm(forward)
        # forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward

        # task = np.pi * 30 / 180
        direction_cam = gt_nor[x, y, :3]
        direction_cam /= np.linalg.norm(direction_cam)
        direction_world = cam.get_metadata()['mat44'][:3, :3] @ direction_cam

        # compute final pose
        # here, RL give the action(qpos), then caculate the rotmat(SE3)
        if args.up_norm_dir:  # start up: norm
            action_direction_cam = -direction_cam
        else:
            if not args.use_random_up:
                raise ValueError
            action_direction_cam = np.random.randn(3).astype(np.float32)
            action_direction_cam /= np.linalg.norm(action_direction_cam)
            while action_direction_cam @ direction_cam > args.up_norm_thresh: # degree distance to normal
                action_direction_cam = np.random.randn(3).astype(np.float32)
                action_direction_cam /= np.linalg.norm(action_direction_cam)
        action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam
        up = np.array(action_direction_world, dtype=np.float32)  # up = action_direction_world
        forward = np.random.randn(3).astype(np.float32)
        forward /= np.linalg.norm(forward)
        while abs(up @ forward) > 0.99:
            forward = np.random.randn(3).astype(np.float32)
            forward /= np.linalg.norm(forward)
        left = np.cross(up, forward)
        left /= np.linalg.norm(left)  # get unit vector
        forward = np.cross(left, up)
        forward /= np.linalg.norm(forward)
        forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward

        rotmat = np.eye(4).astype(np.float32)
        rotmat[:3, 0] = forward
        rotmat[:3, 1] = left
        rotmat[:3, 2] = up

        start_rotmat = np.array(rotmat, dtype=np.float32)
        if args.use_direction_world:
            start_rotmat[:3, 3] = position_world - (-direction_world) * 0.15  # add displacement(lase column)
            start_pose = Pose().from_transformation_matrix(start_rotmat)
            start_gripper_root_position = position_world - (-direction_world) * 0.15
        else:
            start_rotmat[:3, 3] = position_world - action_direction_world * 0.15  # add displacement(lase column)
            start_pose = Pose().from_transformation_matrix(start_rotmat)
            start_gripper_root_position = position_world - action_direction_world * 0.15

        end_rotmat = start_rotmat.copy()
        end_rotmat[:3, 3] = position_world - direction_world * 0.1

        if robot_loaded == 0:
            robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))
            robot_loaded = 1
        else:
            robot.load_gripper(robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))
        env.end_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)

        state_RL = []
        ep_r = 0  # cumulated reward
        reward, final_reward = 0, 0
        final_distance, tot_guidance, tot_penalty = 100, 0, 0

        out_info = dict()
        out_info['task'] = task_degree
        out_info['open_door'] = args.open_door
        out_info['random_seed'] = args.random_seed
        out_info['pixel_locs'] = [int(x), int(y)]
        out_info['target_object_part_actor_id'] = env.target_object_part_actor_id
        out_info['target_object_part_joint_id'] = env.target_object_part_joint_id
        if env.target_object_part_joint_type == ArticulationJointType.REVOLUTE:
            out_info['target_object_part_joint_type'] = "REVOLUTE"
        elif env.target_object_part_joint_type == ArticulationJointType.PRISMATIC:
            out_info['target_object_part_joint_type'] = 'PRISMATIC'
        else:
            out_info['target_object_part_joint_type'] = str(env.target_object_part_joint_type)
        out_info['direction_camera'] = direction_cam.tolist()
        flog.write('Direction Camera: %f %f %f\n' % (direction_cam[0], direction_cam[1], direction_cam[2]))
        out_info['direction_world'] = direction_world.tolist()
        flog.write('Direction World: %f %f %f\n' % (direction_world[0], direction_world[1], direction_world[2]))
        flog.write('mat44: %s\n' % str(cam.get_metadata()['mat44']))
        out_info['gripper_direction_camera'] = action_direction_cam.tolist()
        out_info['gripper_direction_world'] = action_direction_world.tolist()
        out_info['position_cam'] = position_cam.tolist()
        out_info['position_world'] = position_world.tolist()
        out_info['gripper_forward_direction_world'] = forward.tolist()
        out_info['gripper_forward_direction_camera'] = forward_cam.tolist()

        out_info['camera_metadata'] = cam.get_metadata_json()

        out_info['object_state'] = target_part_state
        out_info['joint_angles'] = joint_angles
        out_info['joint_angles_lower'] = env.joint_angles_lower
        out_info['joint_angles_upper'] = env.joint_angles_upper
        out_info['start_rotmat_world'] = start_rotmat.tolist()
        out_info['shape_id'] = shape_id
        out_info['cat'] = selected_cat

        # move back
        robot.robot.set_root_pose(start_pose)
        robot.close_gripper()
        env.render()

        # activate contact checking
        env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)

        if not args.no_gui:
            ### wait to start
            env.wait_to_start()
            pass

        start_target_part_qpos = env.get_target_part_qpos()
        try:
            robot.move_to_target_pose(end_rotmat, 2000)
        except Exception:
            env.scene.remove_articulation(env.object)
            env.scene.remove_articulation(robot.robot)
            flog.close()
            continue
        robot.wait_n_steps(1000)
        end_target_part_qpos = env.get_target_part_qpos()
        radian_dis = radian2degree(end_target_part_qpos - start_target_part_qpos)

        if args.open_door:
            if radian_dis < 2:
                env.scene.remove_articulation(env.object)
                env.scene.remove_articulation(robot.robot)
                flog.close()
                continue
        else:
            if radian_dis > -2:
                env.scene.remove_articulation(env.object)
                env.scene.remove_articulation(robot.robot)
                flog.close()
                continue
        env.scene.remove_articulation(robot.robot)

        env.set_object_joint_angles(joint_angles)
        env.render()

        robot.load_gripper(robot_urdf_fn, robot_material)
        env.end_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)
        robot.robot.set_root_pose(start_pose)
        env.render()
        env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)

        ### main steps
        out_info['start_target_part_qpos'] = env.get_target_part_qpos()
        init_target_part_qpos = env.get_target_part_qpos()

        target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
        position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1

        num_steps = args.num_steps   # waypoints = num_step + 1
        out_info["num_steps"] = num_steps + 1
        waypoints, dense_waypoints, waypoints_qpos = [], [], []

        success = True
        success_traj = False
        transition_list = []

        initial_gripper_finger_position = position_world - 0.02 * action_direction_world

        far_away = False
        try:
            state_RL.append(env.get_target_part_qpos())
            state_RL.append((init_target_part_qpos - task) - env.get_target_part_qpos())
            state_RL.append(-1 * task)
            if args.state_initial_position:
                state_RL.extend(start_gripper_root_position.tolist())
            state_RL.extend(position_world.tolist())     # contact point
            state_RL.extend(initial_gripper_finger_position.tolist())    # gripper
            state_RL.extend(robot.robot.get_qpos().tolist())
            if args.state_axes:
                state_RL.extend([joint_axes[axes_dir]])
            if args.state_axes_all:
                state_RL.extend(joint_axes)
            if args.state_initial_dir:
                state_RL.extend(up.tolist())
                state_RL.extend(forward.tolist())
                state_RL.extend(left.tolist())
            if args.state_initial_up_dir:
                state_RL.extend(up.tolist())
            if args.state_joint_origins:
                state_RL.extend(state_joint_origins)
            if args.state_ctpt_dis_to_joint:
                state_RL.append(state_ctpt_dis_to_joint)
            cur_state = torch.FloatTensor(state_RL).view(1, -1).to(device)   # batch_size = 1

            succ_images = []
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose*255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            succ_images.append(fimg)
            for step_idx in range(num_steps):
                with torch.no_grad():
                    next_qpos = td3.policy_net.get_action(cur_state, deterministic=DETERMINISTIC, explore_noise_scale=0)
                waypoint = next_qpos.detach().cpu().numpy()

                # get rotmat and move
                final_rotmat = start_rotmat.copy()
                if args.pred_residual_root_qpos:
                    final_rotmat[:3, 3] += waypoint[0] * forward + waypoint[1] * left + waypoint[2] * up
                if args.pred_residual_world_xyz:
                    final_rotmat[0, 3] += waypoint[0]
                    final_rotmat[1, 3] += waypoint[1]
                    final_rotmat[2, 3] += waypoint[2]
                if args.pred_world_xyz:
                    final_rotmat[0, 3] = waypoint[0]
                    final_rotmat[1, 3] = waypoint[1]
                    final_rotmat[2, 3] = waypoint[2]
                if args.wp_rot:
                    try:
                        r = R.from_euler('XYZ', [waypoint[3], waypoint[4], waypoint[5]], degrees=False)
                        final_rotmat[:3, :3] = final_rotmat[:3, :3] @ r.as_matrix()
                    except Exception:
                        success = False
                        break
                try:
                    if args.vis_gif:
                        imgs, cur_waypoints = robot.move_to_target_pose(final_rotmat, 2500, cam=cam, vis_gif=True, vis_gif_interval=500, visu=True)
                        succ_images.extend(imgs)
                    else:
                        cur_waypoints = robot.move_to_target_pose(final_rotmat, 2500, cam=cam, vis_gif=False, visu=True)
                    if args.vis_gif:
                        imgs, cur_waypoints2 = robot.wait_n_steps(1000, visu=True, vis_gif=True, vis_gif_interval=200, cam=cam)
                        succ_images.extend(imgs)
                    else:
                        cur_waypoints2 = robot.wait_n_steps(1000, visu=True)
                except Exception:
                    success = False
                    print("Exception")
                    break
                if not success:
                    break
                target_part_trans = env.get_target_part_pose().to_transformation_matrix()   # world coordinate -> target part coordinate transformation matrix
                now_contact_point_position = target_part_trans @ position_local_xyz1    # contact point in world coordinate
                now_contact_point_position = now_contact_point_position[:3]

                now_qpos = robot.robot.get_qpos().tolist()
                now_gripper_root_position = position_world - 0.15 * action_direction_world + now_qpos[0] * forward + now_qpos[1] * left + now_qpos[2] * up

                r = R.from_euler('XYZ', [now_qpos[3], now_qpos[4], now_qpos[5]], degrees=False)
                now_rotmat = final_rotmat[:3, :3] @ r.as_matrix()
                now_up = now_rotmat[:3, 2]
                now_gripper_finger_position = now_gripper_root_position + now_up * 0.13

                # update state
                state_ = []
                state_.append(env.get_target_part_qpos())
                state_.append((init_target_part_qpos - task) - env.get_target_part_qpos())
                state_.append(-1 * task)
                if args.state_initial_position:
                    state_.extend(start_gripper_root_position.tolist())
                state_.extend(now_contact_point_position.tolist())     # contact point
                state_.extend(now_gripper_finger_position.tolist())    # gripper
                state_.extend(robot.robot.get_qpos().tolist())
                if args.state_axes:
                    state_.extend([joint_axes[axes_dir]])
                if args.state_axes_all:
                    state_.extend(joint_axes)
                if args.state_initial_dir:
                    state_.extend(up.tolist())
                    state_.extend(forward.tolist())
                    state_.extend(left.tolist())
                if args.state_initial_up_dir:
                    state_.extend(up.tolist())
                if args.state_joint_origins:
                    state_.extend(state_joint_origins)
                if args.state_ctpt_dis_to_joint:
                    state_.append(state_ctpt_dis_to_joint)
                next_state = torch.FloatTensor(state_).view(1, -1).to(device)

                ''' calculate reward  (use radian) '''
                reward = 0
                stop = False
                distance = np.abs((init_target_part_qpos - task) - env.get_target_part_qpos())
                final_distance = radian2degree(distance)

                # guidance reward
                reward += args.guidance_reward * (np.abs(cur_state[0][1].detach().cpu().numpy()) - np.abs(next_state[0][1].detach().cpu().numpy()))
                tot_guidance += args.guidance_reward * (np.abs(cur_state[0][1].detach().cpu().numpy()) - np.abs(next_state[0][1].detach().cpu().numpy()))

                gripper_ctpt_distance = np.linalg.norm(now_contact_point_position - now_gripper_finger_position)
                # penalty if gripper far from contact point, (add extra penalty)
                reward -= 150 * gripper_ctpt_distance
                tot_penalty -= 150 * gripper_ctpt_distance
                if gripper_ctpt_distance > threshold_gripper_distance:
                    reward -= 300
                    tot_penalty -= 300
                    far_away = True
                    print('gripper is far away from contact point')
                    break  # discard the data if gripper is far away.
                print("gripper_ctpt_dis:", gripper_ctpt_distance)

                waypoints_qpos.append(robot.robot.get_qpos().tolist())
                waypoints.append(waypoint.tolist())
                dense_waypoints.extend(cur_waypoints)
                dense_waypoints.extend(cur_waypoints2)

                if distance < degree2radian(abs(task_degree) * 0.1):
                    reward += args.success_reward
                    stop = True
                    end_step = step_idx + 1
                    tot_done_epoch += 1
                    # export SUCC GIF Image
                    if args.vis_gif:
                        try:
                            imageio.mimsave(os.path.join(result_succ_dir, '%d_%d_%.3f_%.3f_%.3f_%.3f_%d.gif' % (tot_done_epoch, idx_process, task_degree, radian2degree(init_target_part_qpos), radian2degree(joint_angle_upper), waypoint[0], step_idx+1)), succ_images)
                        except:
                            pass

                elif step_idx == num_steps - 1:
                    stop = True
                    end_step = step_idx + 1
                    if tot_fail_epoch < tot_done_epoch:
                        tot_fail_epoch += 1
                        if args.vis_gif:
                            try:
                                imageio.mimsave(os.path.join(result_fail_dir, '%d_%d_%.3f_%.3f_%.3f_%.3f_%.3f_%d.gif' % (tot_fail_epoch, idx_process, task_degree, radian2degree(init_target_part_qpos), radian2degree(joint_angle_upper), radian2degree(distance), waypoint[0], step_idx+1)), succ_images)
                            except:
                                pass

                # store (s,a,r)
                if stop and distance < degree2radian(abs(task_degree) * 0.1):
                    success_traj = True

                ep_r += reward
                sar = [cur_state.view(-1).cpu().detach().numpy(), next_qpos.view(-1).cpu().detach().numpy(), reward, next_state.view(-1).cpu().detach().numpy(), stop]
                transition_list.append(sar)
                cur_state = next_state.clone().detach()

                if stop:
                    break

        except ContactError:
            success = False

        actual_task = radian2degree(init_target_part_qpos - env.get_target_part_qpos())
        out_info['actual_task'] = actual_task
        out_info['distance_to_closeDoor'] = radian2degree(env.get_target_part_qpos())
        target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
        position_world_xyz1_end = target_link_mat44 @ position_local_xyz1
        flog.write('touch_position_world_xyz_start: %s\n' % str(position_world_xyz1))
        flog.write('touch_position_world_xyz_end: %s\n' % str(position_world_xyz1_end))
        out_info['touch_position_world_xyz_start'] = position_world_xyz1[:3].tolist()
        out_info['touch_position_world_xyz_end'] = position_world_xyz1_end[:3].tolist()

        # close the file
        env.scene.remove_articulation(robot.robot)

        if success:
            out_info['result'] = 'VALID'
            out_info['final_target_part_qpos'] = env.get_target_part_qpos()
            out_info['part_motion'] = out_info['final_target_part_qpos'] - out_info['start_target_part_qpos']
            out_info['part_motion_degree'] = out_info['part_motion'] * 180.0 / 3.1415926535
        else:
            out_info['result'] = 'INVALID'
            if env.contact_error:
                out_info['result'] = 'CONTACT_ERROR'
            out_info['part_motion'] = 0.0
            out_info['part_motion_degree'] = 0.0
            print('contact_error')
            # flog.close()
            # env.scene.remove_articulation(env.object)
            # continue

        out_info['waypoints'] = waypoints
        out_info['waypoints_qpos'] = waypoints_qpos
        out_info['dense_waypoints'] = dense_waypoints

        if args.no_gui:
            pass
        else:
            if success:
                print('[Successful Interaction] Done. Ctrl-C to quit.')
            else:
                print('[Unsuccessful Interaction] invalid gripper-object contact.')

        final_reward = reward
        traj_info = [final_distance, ep_r, final_reward, tot_penalty, tot_guidance]
        cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]
        if (not far_away) and (len(waypoints) != 0):
            transition_Q.put([idx_process, transition_list, waypoints, success_traj, traj_info, out_info, cam_XYZA_list, gt_all_movable_link_mask, gt_movable_link_mask])

        flog.close()
        env.scene.remove_articulation(env.object)

trial_id = args.trial_id
primact_type = args.primact_type

out_dir = os.path.join(args.out_dir,
                       'RL_%s_%d' % (args.action_type, trial_id))

print('out_dir: ', out_dir)
if not args.novel_cat:
    result_succ_dir = os.path.join(out_dir, 'result_succ_imgs_collect')
    result_fail_dir = os.path.join(out_dir, 'result_fail_imgs_collect')
else:
    result_succ_dir = os.path.join(out_dir, 'result_succ_imgs_collect_novel')
    result_fail_dir = os.path.join(out_dir, 'result_fail_imgs_collect_novel')
if not os.path.exists(result_succ_dir):
    os.mkdir(result_succ_dir)
if not os.path.exists(result_fail_dir):
    os.mkdir(result_fail_dir)

trans_q = mp.Queue()

degree_record = [0 for i in range(8)]
degree_uplimit = [args.degree_uplimit for i in range(8)]
num_closeDoor = 0
num_contactError = 0

for idx_process in range(args.num_processes):
    p = mp.Process(target=run_an_RL, args=(idx_process, args, trans_q))
    p.start()

device = args.device
if not torch.cuda.is_available():
    device = "cpu"

EP_MAX = 100000
hidden_dim = 512
DETERMINISTIC = True  # DDPG: deterministic policy gradient
threshold_gripper_distance = args.threshold
num_steps = args.num_steps

# task = np.pi * args.task / 180
pos_range = args.pos_range
rot_range = np.pi * args.rot_degree / 180
action_range = torch.tensor([pos_range, pos_range, pos_range, rot_range, rot_range, rot_range]).to(device)
action_dim = 6
state_dim = 1 + 1 + 1 + 3 + 3 + 8  # cur_obj_qpos, dis_to_target, cur_gripper_info, cur_step_idx, final_task(degree), contact_point_xyz, gripper_xyz
if args.state_initial_position:
    state_dim += 3
if args.state_initial_dir:
    state_dim += 9
if args.state_initial_up_dir:
    state_dim += 3
if args.state_joint_origins:
    state_dim += 3
if args.state_ctpt_dis_to_joint:
    state_dim += 1
if args.state_axes:
    state_dim += 1
if args.state_axes_all:
    state_dim += 3
# state
# 0     : door's current qpos
# 1     : distance to task
# 2     : task
# 3-5   : start_gripper_root_position
# 6-8   : contact_point_position_world
# 9-11  : gripper_finger_position
# 12-19 : gripper_qpos
# 20-28 : up, forward, left
#       : up
# 29-31 : joint_origins
# 32    : state_ctpt_dis_to_joint
# 33-37 : step_idx


saved_dir = os.path.join(out_dir, args.out_folder, 'allShape_%d_%d' % (trial_id, args.dataset_id))
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)

epoch = -1
record = [0 for idx in range(100)]
# unsuccess, success + high critic_score, success + low critic_score
best_acc = 0.0

t0 = time.time()
while True:
    if not trans_q.empty():
        epoch += 1

        idx_process, transition_list, waypoints, success_traj, traj_info, out_info, cam_XYZA_list, gt_all_movable_link_mask, gt_movable_link_mask = trans_q.get()
        out_info['epoch'] = epoch
        cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA = cam_XYZA_list
        final_distance, ep_r, final_reward, tot_penalty, tot_guidance = traj_info
        len_episode = len(transition_list)

        if success_traj:
            record[epoch % 100] = 1
        else:
            record[epoch % 100] = 0

        task_degree = out_info['task']
        actual_task = out_info['actual_task']
        distance_to_closeDoor = out_info['distance_to_closeDoor'] # degree
        result = out_info['result']
        valid = False
        contact_error = False
        if result == 'CONTACT_ERROR':
            contact_error = True
        if result == 'VALID':
            valid = True

        save_result = False

        positive_actual_task = actual_task

        if valid and positive_actual_task >= args.task_lower and positive_actual_task <= args.task_upper and degree_record[
            int(positive_actual_task // 10) - 1] < degree_uplimit[int(positive_actual_task // 10) - 1] and len(waypoints) != 0:
            # check if the door is closed
            if distance_to_closeDoor > 2:
                save_result = True
            else:
                if num_closeDoor < args.closeDoor_uplimit:
                    save_result = True
                    num_closeDoor += 1
                else:
                    print("close door")

        if save_result:
            with open(os.path.join(saved_dir, 'result_%d.json' % epoch), 'w') as fout:
                json.dump(out_info, fout)
            save_h5(os.path.join(saved_dir, 'cam_XYZA_%d.h5' % epoch), [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                                                                      (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                                                                      (cam_XYZA_pts.astype(np.float32), 'pc',
                                                                       'float32')])
            Image.fromarray((gt_movable_link_mask > 0).astype(np.uint8) * 255).save(
                os.path.join(saved_dir, 'interaction_mask_%d.png' % epoch))

            degree_record[int(positive_actual_task // 10) - 1] += 1


        # save contact error
        if contact_error:
            with open(os.path.join(saved_dir, 'result_%d.json' % epoch), 'w') as fout:
                json.dump(out_info, fout)
            save_h5(os.path.join(saved_dir, 'cam_XYZA_%d.h5' % epoch), [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                                                                      (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                                                                      (cam_XYZA_pts.astype(np.float32), 'pc',
                                                                       'float32')])
            Image.fromarray((gt_movable_link_mask > 0).astype(np.uint8) * 255).save(
                os.path.join(saved_dir, 'interaction_mask_%d.png' % epoch))
            num_contactError += 1

        accuracy = sum(record) / len(record)
        if best_acc < accuracy + 0.001:
            best_acc = accuracy

        print(
            'Epo: {}  | Dis: {:.4f}  | EP_R: {:.4f}  | F_R: {:.4f}  | Pen: {:.4f}  | Acc: {:.4f} | Tim: {:.4f}'.format(
                epoch, final_distance, ep_r, final_reward, tot_penalty, accuracy,
                time.time() - t0
            )
        )
        print('actual_task: ', actual_task, 'if_saved: ', save_result)
        print('task degree distribution: ', degree_record)
        print('num_contactError: ', num_contactError, '\n')
        print("tot:", sum(degree_record))

        t0 = time.time()



