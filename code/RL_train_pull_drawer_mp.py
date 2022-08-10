import os
import sys
import shutil
import numpy as np
from PIL import Image
from utils import radian2degree
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
from tensorboardX import SummaryWriter
from scipy.spatial.transform import Rotation as R
import imageio
import multiprocessing as mp

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--category', type=str)
parser.add_argument('--primact_type', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--trial_id', type=int, default=0, help='trial id')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_degree', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--every_bonus', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_initial_position', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_initial_dir', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_initial_up_dir', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_joint_origins', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_ctpt_dis_to_joint', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--pn_feat', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--train_pn', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--pn_feat_dim', type=int, default=-1)
parser.add_argument('--pnpp_ckpt_path', type=str, default='xxx')
parser.add_argument('--pred_world_xyz', type=int, default=0)
parser.add_argument('--pred_residual_world_xyz', type=int, default=0)
parser.add_argument('--pred_residual_root_qpos', type=int, default=1)
parser.add_argument('--up_norm_dir', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--final_dist', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--replay_buffer_size', type=int, default=5e5)
parser.add_argument('--pos_range', type=float, default=0.5)
parser.add_argument('--explore_noise_scale', type=float, default=0.01)
parser.add_argument('--eval_noise_scale', type=float, default=0.01)
parser.add_argument('--noise_decay', type=float, default=0.8)
parser.add_argument('--guidance_reward', type=float, default=0.0)
parser.add_argument('--decay_interval', type=int, default=100)
parser.add_argument('--q_lr', type=float, default=3e-4)
parser.add_argument('--policy_lr', type=float, default=3e-4)
parser.add_argument('--threshold', type=float, default=0.1 )
parser.add_argument('--task_upper', type=float, default=30)
parser.add_argument('--task_lower', type=float, default=30)
parser.add_argument('--success_reward', type=int, default=10)
parser.add_argument('--target_margin', type=float, default=2)
parser.add_argument('--HER_move_margin', type=float, default=2)
parser.add_argument('--target_part_state', type=str, default='random-middle')
parser.add_argument('--num_steps', type=int, default=4)
parser.add_argument('--with_step', type=int, default=1)
parser.add_argument('--update_itr2', type=int, default=2)
parser.add_argument('--early_stop', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--use_HER', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--HER_only_success', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--HER_only_attach', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--sample_num', type=int, default=3)
parser.add_argument('--wp_rot', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--use_direction_world', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--up_norm_thresh', type=float, default=0)
parser.add_argument('--use_random_up', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--all_shape', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_axes', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--open_door', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--rot_range', type=float, default=45)

parser.add_argument('--RL_pretrained_id', type=int, default=0, help='trial id')
parser.add_argument('--RL_ckpt_dir', type=str)
parser.add_argument('--RL_load_epoch', type=int)
parser.add_argument('--num_processes', type=int, default=1)
parser.add_argument('--save_td3_epoch', type=int, default=10)
parser.add_argument('--task_succ_margin', type=float, default=0.10)

parser.add_argument('--action_type', type=str, default="xxx")

args = parser.parse_args()

def run_an_RL(idx_process, args, transition_Q, epoch_Q, decay_Q):
    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"

    np.random.seed(random.randint(1, 1000) + idx_process)
    random.seed(random.randint(1, 1000) + idx_process)

    train_shape_list, val_shape_list = [], []
    train_file_dir = "../stats/train_VAT_train_data_list.txt"
    val_file_dir = "../stats/train_VAT_test_data_list.txt"
    all_shape_list = []

    with open(train_file_dir, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            if cat != 'StorageFurniture':
                continue
            train_shape_list.append(shape_id)
            all_shape_list.append(shape_id)

    with open(val_file_dir, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            if cat != 'StorageFurniture':
                continue
            val_shape_list.append(shape_id)
            all_shape_list.append(shape_id)

    len_train_shape_list = len(train_shape_list)
    len_all_shape_list = len(all_shape_list)

    EP_MAX = 1000000
    hidden_dim = 512
    policy_target_update_interval = 3 # delayed update for the policy network and target networks
    DETERMINISTIC = True  # DDPG: deterministic policy gradient
    explore_noise_scale = args.explore_noise_scale
    eval_noise_scale = args.eval_noise_scale
    if args.wp_rot:
        explore_noise_scale = torch.tensor([explore_noise_scale, explore_noise_scale, explore_noise_scale, explore_noise_scale * 3, explore_noise_scale * 3, explore_noise_scale * 3])
        eval_noise_scale = torch.tensor([eval_noise_scale, eval_noise_scale, eval_noise_scale, eval_noise_scale * 3, eval_noise_scale * 3, eval_noise_scale * 3])
    reward_scale = 1.
    noise_decay = args.noise_decay
    decay_interval = args.decay_interval
    threshold_gripper_distance = args.threshold

    # task = np.pi * args.task / 180
    pos_range = args.pos_range
    rot_range = np.pi * args.rot_range / 180
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

    replay_buffer_size = args.replay_buffer_size
    replay_buffer = ReplayBuffer(replay_buffer_size)
    td3 = TD3(replay_buffer, state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, policy_target_update_interval=policy_target_update_interval, action_range=action_range, q_lr=args.q_lr, policy_lr=args.policy_lr, device=device, pred_world_xyz=args.pred_world_xyz, wp_rot=args.wp_rot).to(device)


    trial_id = args.trial_id
    primact_type = args.primact_type

    out_dir = os.path.join(args.out_dir,
                           'RL_%s_%d' % (args.action_type, trial_id))

    task_succ_margin = args.task_succ_margin

    # setup env
    print("creating env")
    env = Env(show_gui=(not args.no_gui))
    print("env creared")


    ### viz the EE gripper position
    # setup robot
    robot_urdf_fn = './robots/panda_gripper.urdf'
    robot_material = env.get_material(4, 4, 0.01)

    target_part_state = args.target_part_state
    if args.state_degree:
        args.guidance_reward *= (np.pi / 180)

    if args.pred_world_xyz + args.pred_residual_world_xyz + args.pred_residual_root_qpos != 1:
        raise ValueError

    robot_loaded = 0
    tot_done_epoch = 1
    tot_fail_epoch = 0
    object_material = env.get_material(4, 4, 0.01)
    cam = None

    prev_epoch_qsize = 0
    prev_decay_qsize = 0
    save_td3_epoch = int(args.save_td3_epoch)

    for epoch in range(EP_MAX):
        now_epoch_qsize = epoch_Q.qsize()
        if now_epoch_qsize > prev_epoch_qsize:
            prev_epoch_qsize = now_epoch_qsize
            td3.load_model(path=os.path.join(out_dir, 'td3_%d' % ((now_epoch_qsize - 1) * save_td3_epoch)))
            td3.train()
        now_decay_qsize = decay_Q.qsize()
        if now_decay_qsize > prev_decay_qsize:
            prev_decay_qsize = now_decay_qsize
            explore_noise_scale *= noise_decay
            eval_noise_scale *= noise_decay
        torch.cuda.empty_cache()

        if cam is None:
            cam = Camera(env, random_position=True, restrict_dir=True)
        else:
            cam.change_pose(random_position=True, restrict_dir=True)
        if not args.no_gui:
            env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)

        torch.cuda.empty_cache()

        flog = open(os.path.join(out_dir, '%s_log.txt' % str(idx_process)), 'a')
        env.flog = flog

        if args.all_shape:
            shape_id = all_shape_list[random.randint(0, len_all_shape_list - 1)]
        else:
            shape_id = train_shape_list[random.randint(0, len_train_shape_list - 1)]

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

        idx = np.random.randint(len(xs))
        x, y = xs[idx], ys[idx]
        target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
        env.set_target_object_part_actor_id(target_part_id)
        tot_trial = 0
        while tot_trial < 50 and (env.target_object_part_joint_type != ArticulationJointType.PRISMATIC):
            idx = np.random.randint(len(xs))
            x, y = xs[idx], ys[idx]
            target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
            env.set_target_object_part_actor_id(target_part_id)
            tot_trial += 1
        if (env.target_object_part_joint_type != ArticulationJointType.PRISMATIC):
            env.scene.remove_articulation(env.object)
            flog.close()
            continue
        target_part_joint_idx = env.target_object_part_joint_id

        joint_angle_lower = env.joint_angles_lower[target_part_joint_idx]
        joint_angle_upper = env.joint_angles_upper[target_part_joint_idx]
        # joint_angle_lower_degree = radian2degree(joint_angle_lower)
        # joint_angle_upper_degree = radian2degree(joint_angle_upper)
        task_upper = min(joint_angle_upper - joint_angle_lower, args.task_upper)

        task_lower = args.task_lower
        # task_lower = max(args.task_lower, radian2degree(target_part_qpos))    # task = np.pi * 30 / 180
        task_radian = random.random() * (task_upper - task_lower) + task_lower

        joint_angles = env.update_joint_angle(joint_angles, target_part_joint_idx, target_part_state, task_radian, push=False, pull=True, drawer=True)
        env.set_object_joint_angles(joint_angles)
        env.render()
        task_radian = - task_radian
        task = task_radian

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
        if axes_dir != 0:
            env.scene.remove_articulation(env.object)
            flog.close()
            continue

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
        if args.up_norm_dir:
            action_direction_cam = -direction_cam
        else:
            if not args.use_random_up:
                raise ValueError
            action_direction_cam = np.random.randn(3).astype(np.float32)
            action_direction_cam /= np.linalg.norm(action_direction_cam)
            while action_direction_cam @ direction_cam > args.up_norm_thresh:
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
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if args.use_direction_world:
            start_rotmat[:3, 3] = position_world - (-direction_world) * 0.15  # add displacement(lase column)
            start_pose = Pose().from_transformation_matrix(start_rotmat)
            start_gripper_root_position = position_world - (-direction_world) * 0.15
        else:
            start_rotmat[:3, 3] = position_world - action_direction_world * 0.15  # add displacement(lase column)
            start_pose = Pose().from_transformation_matrix(start_rotmat)
            start_gripper_root_position = position_world - action_direction_world * 0.15

        end_rotmat = start_rotmat.copy()
        end_rotmat[:3, 3] = position_world - action_direction_world * 0.08

        if robot_loaded == 0:
            robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))
            robot_loaded = 1
        else:
            robot.load_gripper(robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))
        env.end_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)

        state_RL = []
        ep_r = 0  # cumulated reward
        reward, final_reward = 0, 0
        bonus, penalty = 0, 0
        final_distance, tot_guidance, tot_penalty = 100, 0, 0

        out_info = dict()
        out_info['task'] = task_radian
        out_info['shape_id'] = shape_id
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

        # move back
        robot.robot.set_root_pose(start_pose)
        env.render()

        # activate contact checking
        env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)

        init_success = True
        success_grasp = False
        try:
            robot.open_gripper()
            robot.move_to_target_pose(end_rotmat, 3000)
            robot.wait_n_steps(2000)
            robot.close_gripper()
            robot.wait_n_steps(600)
            now_qpos = robot.robot.get_qpos().tolist()
            finger1_qpos = now_qpos[-1]
            finger2_qpos = now_qpos[-2]
            if finger1_qpos + finger2_qpos > 0.01:
                success_grasp = True
        except Exception:
            init_success = False
        if not (success_grasp and init_success):
            flog.close()
            env.scene.remove_articulation(env.object)
            env.scene.remove_articulation(robot.robot)
            continue

        if not args.no_gui:
            ### wait to start
            env.wait_to_start()
            pass

        ### main steps
        out_info['start_target_part_qpos'] = env.get_target_part_qpos()
        init_target_part_qpos = env.get_target_part_qpos()

        target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
        position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1

        num_steps = args.num_steps   # waypoints = num_step + 1
        out_info["num_steps"] = num_steps + 1
        step_one_hot = np.eye(num_steps + 1)
        waypoints = []

        success = True
        success_traj = False
        transition_list = []

        gripper_finger_position = position_world - 0.02 * action_direction_world

        try:
            if args.state_degree:
                state_RL.append(radian2degree(env.get_target_part_qpos()))
                state_RL.append(radian2degree((init_target_part_qpos - task) - env.get_target_part_qpos()))
                state_RL.append(radian2degree(-1 * task))
            else:
                state_RL.append(env.get_target_part_qpos())
                state_RL.append((init_target_part_qpos - task) - env.get_target_part_qpos())
                state_RL.append(-1 * task)
            if args.state_initial_position:
                state_RL.extend(start_gripper_root_position.tolist())
            state_RL.extend(position_world.tolist())     # contact point
            state_RL.extend(gripper_finger_position.tolist())    # gripper
            state_RL.extend(robot.robot.get_qpos().tolist())
            if args.state_axes:
                state_RL.extend([joint_axes[axes_dir]])
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
                next_qpos = td3.policy_net.get_action(cur_state, deterministic=DETERMINISTIC, explore_noise_scale=explore_noise_scale)
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
                    imgs = robot.move_to_target_pose(final_rotmat, 3000, cam=cam, vis_gif=True, vis_gif_interval=500)
                    succ_images.extend(imgs)
                    robot.wait_n_steps(600)
                    robot.close_gripper()
                    robot.wait_n_steps(400)
                except Exception:
                    success = False
                    break
                if not success:
                    break
                waypoints.append(waypoint.tolist())
                target_part_trans = env.get_target_part_pose().to_transformation_matrix()
                now_contact_point_position = target_part_trans @ position_local_xyz1
                now_contact_point_position = now_contact_point_position[:3]

                now_qpos = robot.robot.get_qpos().tolist()
                now_gripper_position = gripper_finger_position + now_qpos[0] * forward + now_qpos[1] * left + now_qpos[2] * up

                # update state
                state_ = []
                if args.state_degree:
                    state_.append(radian2degree(env.get_target_part_qpos()))
                    state_.append(radian2degree((init_target_part_qpos - task) - env.get_target_part_qpos()))
                    state_.append(radian2degree(-1 * task))
                else:
                    state_.append(env.get_target_part_qpos())
                    state_.append((init_target_part_qpos - task) - env.get_target_part_qpos())
                    state_.append(-1 * task)
                if args.state_initial_position:
                    state_.extend(start_gripper_root_position.tolist())
                state_.extend(now_contact_point_position.tolist())     # contact point
                state_.extend(now_gripper_position.tolist())    # gripper
                state_.extend(robot.robot.get_qpos().tolist())
                if args.state_axes:
                    state_.extend([joint_axes[axes_dir]])
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
                # penalty = 0
                stop = False
                distance = np.abs((init_target_part_qpos - task) - env.get_target_part_qpos())
                final_distance = radian2degree(distance)

                # guidance reward
                reward += args.guidance_reward * (np.abs(cur_state[0][1].detach().cpu().numpy()) - np.abs(next_state[0][1].detach().cpu().numpy()))    # 10 degree, reward ~ 0.87
                tot_guidance += args.guidance_reward * (np.abs(cur_state[0][1].detach().cpu().numpy()) - np.abs(next_state[0][1].detach().cpu().numpy()))

                # penalty if gripper far from contact point, (add extra penalty)
                reward -= 150 * np.linalg.norm(now_contact_point_position - now_gripper_position)
                tot_penalty -= 150 * np.linalg.norm(now_contact_point_position - now_gripper_position)
                if np.linalg.norm(now_contact_point_position - now_gripper_position) > threshold_gripper_distance:
                    reward -= 300
                    tot_penalty -= 300

                if distance < abs(task_radian) * task_succ_margin:  # bonus
                    reward += args.success_reward
                    bonus += args.success_reward
                    stop = True
                    end_step = step_idx + 1
                    tot_done_epoch += 1
                    # export SUCC GIF Image
                    try:
                        imageio.mimsave(os.path.join(result_succ_dir, '%d_%d_%.3f_%.3f_%.3f_%.3f_%d.gif' % (tot_done_epoch, idx_process, task_radian, radian2degree(init_target_part_qpos), radian2degree(joint_angle_upper), radian2degree(distance), step_idx+1)), succ_images)
                    except:
                        pass

                elif step_idx == num_steps - 1:
                    stop = True
                    end_step = step_idx + 1
                    if tot_fail_epoch < tot_done_epoch * 100:
                        tot_fail_epoch += 1
                        try:
                            imageio.mimsave(os.path.join(result_fail_dir, '%d_%d_%.3f_%.3f_%.3f_%.3f_%d.gif' % (tot_fail_epoch, idx_process, task_radian, radian2degree(init_target_part_qpos), radian2degree(joint_angle_upper), radian2degree(distance), step_idx+1)), succ_images)
                        except:
                            pass

                # store (s,a,r)
                if stop and distance < abs(task_radian) * task_succ_margin:
                    success_traj = True

                ep_r += reward
                sar = [cur_state.view(-1).cpu().detach().numpy(), next_qpos.view(-1).cpu().detach().numpy(), reward, next_state.view(-1).cpu().detach().numpy(), stop]
                transition_list.append(sar)
                cur_state = next_state.clone().detach()

                if stop:
                    break
            final_reward = reward

        except ContactError:
            success = False

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
            flog.close()
            env.scene.remove_articulation(env.object)
            continue

        out_info['waypoints'] = waypoints
        # for idx in range(1000000000000000):
        #     env.render()

        if args.no_gui:
            # close env
            # env.close()
            pass
        else:
            if success:
                pass
                print('[Successful Interaction] Done. Ctrl-C to quit.')
                ### wait forever
                # robot.wait_n_steps(100000000000)
                # env.close()
            else:
                pass
                print('[Unsuccessful Interaction] invalid gripper-object contact.')
                # close env
                # env.close()

        final_reward = reward
        traj_info = [final_distance, ep_r, final_reward, bonus, tot_penalty, tot_guidance]
        # traj_setting = []
        cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]
        transition_Q.put([idx_process, transition_list, waypoints, success_traj, traj_info, out_info, cam_XYZA_list, gt_all_movable_link_mask, gt_movable_link_mask])

        flog.close()
        env.scene.remove_articulation(env.object)

trial_id = args.trial_id
primact_type = args.primact_type

out_dir = os.path.join(args.out_dir,
                       'RL_%s_%d' % (args.action_type, trial_id))

if os.path.exists(out_dir):
    response = input('Out directory "%s" already exists, overwrite? (y/n) ' % out_dir)
    if response != 'y' and response != 'Y':
        sys.exit()
    shutil.rmtree(out_dir)
os.makedirs(out_dir)
print('out_dir: ', out_dir)
result_succ_dir = os.path.join(out_dir, 'result_succ_imgs')
if not os.path.exists(result_succ_dir):
    os.mkdir(result_succ_dir)
result_fail_dir = os.path.join(out_dir, 'result_fail_imgs')
if not os.path.exists(result_fail_dir):
    os.mkdir(result_fail_dir)
json_out_dir = os.path.join(out_dir, "jsons")
if not os.path.exists(json_out_dir):
    os.mkdir(json_out_dir)

trans_q = mp.Queue()
epoch_q = mp.Queue()
decay_q = mp.Queue()

for idx_process in range(args.num_processes):
    p = mp.Process(target=run_an_RL, args=(idx_process, args, trans_q, epoch_q, decay_q))
    p.start()

device = args.device
if not torch.cuda.is_available():
    device = "cpu"

EP_MAX = 100000
update_itr = 1
update_itr2 = args.update_itr2      # HER
batch_size = args.batch_size
hidden_dim = 512
policy_target_update_interval = 3 # delayed update for the policy network and target networks
DETERMINISTIC = True  # DDPG: deterministic policy gradient
explore_noise_scale = args.explore_noise_scale
eval_noise_scale = args.eval_noise_scale
reward_scale = 1.
noise_decay = args.noise_decay
decay_interval = args.decay_interval
threshold_gripper_distance = args.threshold
num_steps = args.num_steps

# task = np.pi * args.task / 180
pos_range = args.pos_range
rot_range = np.pi * 45 / 180
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

# RL
replay_buffer_size = args.replay_buffer_size
replay_buffer = ReplayBuffer(replay_buffer_size)
td3 = TD3(replay_buffer, state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, policy_target_update_interval=policy_target_update_interval, action_range=action_range, q_lr=args.q_lr, policy_lr=args.policy_lr, device=device, pred_world_xyz=args.pred_world_xyz, wp_rot=args.wp_rot).to(device)
# td3.load_model(path=os.path.join(RL_ckpt_dir, 'td3_%d' % args.RL_load_epoch))
td3.train()

# load dataset
data_features = ['pcs', 'pc_pxids', 'pc_movables', 'gripper_img_target', 'gripper_direction_world', 'gripper_forward_direction_world',
                 'result', 'task_motion', 'gt_motion', 'task_waypoints', 'cur_dir', 'shape_id', 'trial_id', 'is_original', 'position_world']

epoch = -1
record = [0 for idx in range(100)]
# unsuccess, success + high critic_score, success + low critic_score
tb_writer = SummaryWriter(os.path.join(out_dir, 'tb_logs'))
num_success = [1 for idx in range(100)]
task_succ_margin = args.task_succ_margin

t0 = time.time()
while True:
    if not trans_q.empty():
        epoch += 1
        if epoch % args.save_td3_epoch == 0:
            td3.save_model(path=os.path.join(out_dir, 'td3_%d' % epoch))
            epoch_q.put(epoch)
        if (epoch % decay_interval == 0) and (epoch != 0):
            explore_noise_scale *= noise_decay
            eval_noise_scale *= noise_decay
            print('decayed', explore_noise_scale, eval_noise_scale)
            decay_q.put(epoch)

        idx_process, transition_list, waypoints, success_traj, traj_info, out_info, cam_XYZA_list, gt_all_movable_link_mask, gt_movable_link_mask = trans_q.get()
        out_info['epoch'] = epoch
        cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA = cam_XYZA_list
        final_distance, ep_r, final_reward, bonus, tot_penalty, tot_guidance = traj_info
        len_episode = len(transition_list)

        if success_traj:
            num_success[idx_process] += 1
            record[epoch % 100] = 1
            # with open(os.path.join(json_out_dir, str(num_success[idx_process]) + "_" + str(idx_process) + '_result.json'), 'w') as fout:
            #     json.dump(out_info, fout)
        else:
            record[epoch % 100] = 0

        # action_direction_world, forward, position_world, task_radian, x, y = traj_setting
        action_direction_world = out_info['gripper_direction_world']
        action_direction_world = np.array(action_direction_world)
        forward = out_info['gripper_forward_direction_world']
        forward = np.array(forward)
        position_world = out_info['position_world']
        task_radian = out_info['task']
        x, y = out_info['pixel_locs']
        for step_idx in range(len_episode):
            sar = transition_list[step_idx]
            cur_state, next_qpos, reward, next_state, stop = sar
            replay_buffer.push(cur_state, next_qpos, reward, next_state, done=stop)
            if len(replay_buffer) > batch_size:
                for i in range(update_itr):
                    loss = td3.update(batch_size, deterministic=DETERMINISTIC, eval_noise_scale=eval_noise_scale, reward_scale=reward_scale)
                    tb_writer.add_scalar('q1_value_loss', loss[1].detach().cpu().numpy(), epoch * num_steps + step_idx)
                    tb_writer.add_scalar('q2_value_loss', loss[2].detach().cpu().numpy(), epoch * num_steps + step_idx)
                    if len(loss) == 4:
                        tb_writer.add_scalar('policy_loss', loss[3].detach().cpu().numpy(), epoch * num_steps + step_idx)

        # HER
        if args.use_HER:
            init_target_part_qpos = out_info['start_target_part_qpos']
            sample_num = args.sample_num
            for i, transition in enumerate(transition_list):
                cur_state = transition[0]
                action = transition[1]
                next_state = transition[3]

                cur_position = cur_state[0]
                next_position = next_state[0]
                if abs(radian2degree(cur_position - next_position)) < args.HER_move_margin:
                    continue
                if args.state_initial_position:
                    now_contact_point_position = next_state[6: 9]
                    now_gripper_position = next_state[9: 12]
                else:
                    now_contact_point_position = next_state[3: 6]
                    now_gripper_position = next_state[6: 9]

                # get new goals
                epi_to_go = transition_list[i:]
                if len(epi_to_go) < sample_num:
                    sample_trans = epi_to_go
                else:
                    sample_trans = random.sample(epi_to_go, sample_num)
                new_goals = [trans[3][0] for trans in sample_trans]

                for new_goal in new_goals:
                    if radian2degree(abs(cur_position - new_goal)) < 3:
                        continue
                    if cur_position > new_goal:
                        continue
                    done = False

                    # guidance reward
                    reward = args.guidance_reward * (np.abs(cur_state[1]) - np.abs(next_state[1]))  # 10 degree, reward ~ 0.87

                    reward -= 150 * np.linalg.norm(now_contact_point_position - now_gripper_position)
                    if np.linalg.norm(now_contact_point_position - now_gripper_position) > threshold_gripper_distance:
                        reward -= 300
                        # print("p2:", -100)

                    new_task = np.array([new_goal - init_target_part_qpos])
                    if abs(next_position - new_goal) < abs(new_task) * task_succ_margin:
                        # bonus
                        reward += args.success_reward
                        done = True

                    new_cur_state = np.concatenate(
                        [np.array([cur_state[0]]), np.array([new_goal - cur_position]), new_task,
                         np.array(cur_state[3:])])
                    new_next_state = np.concatenate(
                        [np.array([next_state[0]]), np.array([new_goal - next_position]), new_task,
                         np.array(next_state[3:])])

                    finish = (cur_state[-2] == 1)
                    if args.HER_only_success:
                        if done:
                            replay_buffer.push(new_cur_state, action, reward, new_next_state, done)
                    else:
                        replay_buffer.push(new_cur_state, action, reward, new_next_state, done)

            # update
            if len(replay_buffer) > batch_size:
                for i in range(update_itr2):
                    loss = td3.update(batch_size, deterministic=DETERMINISTIC, eval_noise_scale=eval_noise_scale, reward_scale=reward_scale)
                    tb_writer.add_scalar('q1_value_loss', loss[1].detach().cpu().numpy(), epoch * num_steps + step_idx)
                    tb_writer.add_scalar('q2_value_loss', loss[2].detach().cpu().numpy(), epoch * num_steps + step_idx)
                    if len(loss) == 4:
                        tb_writer.add_scalar('policy_loss', loss[3].detach().cpu().numpy(), epoch * num_steps + step_idx)

        accuracy = sum(record) / len(record)

        tb_writer.add_scalar('accuracy', accuracy, epoch + 1)

        print(
            'Episode: {}/{}  | Task: {:.4f}  | Final Distance: {:.4f}  | Episode Reward: {:.4f}  | Episode Final Reward: {:.4f}  | Bonus: {:.4f}  | Penalty: {:.4f}  | Guidance: {:.4f}  | Accuracy: {:.4f} | Running Time: {:.4f}'.format(
                epoch, EP_MAX, task_radian, final_distance, ep_r, final_reward, bonus, tot_penalty, tot_guidance, accuracy,
                time.time() - t0
            )
        )
        tb_writer.add_scalar('episode_reward', ep_r, epoch)
        tb_writer.add_scalar('final_reward', final_reward, epoch)
        if final_distance < 90:
            tb_writer.add_scalar('final_distance', final_distance, epoch)
        tb_writer.add_scalar('tot_guidance', tot_guidance, epoch)
        tb_writer.add_scalar('tot_penalty', tot_penalty, epoch)
        tb_writer.add_scalar('accuracy', accuracy, epoch)

        t0 = time.time()



