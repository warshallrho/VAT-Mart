import os
import sys
import shutil
import numpy as np
from PIL import Image
import utils
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
from data_iclr import SAPIENVisionDataset
from train_3d_task_traj_critic_RL import critic_forward
from tensorboardX import SummaryWriter
from scipy.spatial.transform import Rotation as R
import imageio
import multiprocessing as mp
from pointnet2_ops.pointnet2_utils import furthest_point_sample

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--category', type=str, default='None')
parser.add_argument('--cnt_id', type=int, default=0)
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
parser.add_argument('--threshold', type=float, default=0.3)
parser.add_argument('--task_upper', type=float, default=30)
parser.add_argument('--task_lower', type=float, default=30)
parser.add_argument('--success_reward', type=int, default=10)
# parser.add_argument('--target_margin', type=float, default=2)
parser.add_argument('--HER_move_margin', type=float, default=2)
parser.add_argument('--target_part_state', type=str, default='random-middle')
parser.add_argument('--num_steps', type=int, default=4)
parser.add_argument('--update_itr2', type=int, default=2)
parser.add_argument('--early_stop', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--use_HER', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--HER_only_success', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--HER_only_attach', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--sample_num', type=int, default=2)
parser.add_argument('--wp_rot', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--use_direction_world', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--up_norm_thresh', type=float, default=0)
parser.add_argument('--use_random_up', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--all_shape', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_axes', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_axes_all', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_door_dir', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--open_door', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--continue_to_play', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--critic_exp_dir', type=str, default=None, help='critic_exp_dir')


# critic's args
parser.add_argument('--critic_model_version', type=str, help='model def file')
parser.add_argument('--critic_log_dir', type=str, default='/home/username/VAT_Mart/VAT_Data', help='exp logs directory')
parser.add_argument('--critic_exp_suffix', type=int)
parser.add_argument('--critic_load_epoch', type=str, default='best')
parser.add_argument('--critic_batch_size', type=int, default=32)
parser.add_argument('--critic_replay_buffer_size', type=int, default=2048)
parser.add_argument('--critic_feat_dim', type=int, default=128)
parser.add_argument('--critic_lr', type=float, default=0.001)
parser.add_argument('--critic_weight_decay', type=float, default=1e-5)
parser.add_argument('--critic_lr_decay_by', type=float, default=0.9)
parser.add_argument('--critic_lr_decay_every', type=float, default=5000)
parser.add_argument('--critic_update_frequently', action='store_true', default=False, help='no_gui [default: False]')


parser.add_argument('--num_point_per_shape', type=int, default=10000)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--save_critic_epoch', type=int, default=100)

parser.add_argument('--critic_degree_upper', type=float, default=45)
parser.add_argument('--critic_degree_lower', type=float, default=10)
parser.add_argument('--critic_score_threshold', type=float, default=0.5)
parser.add_argument('--critic_update_itr', type=int, default=1)
parser.add_argument('--lbd_critic_penalty', type=int, default=500)
parser.add_argument('--coordinate_system', type=str, default='world', help='world or cam or cambase')
parser.add_argument('--sample_type', type=str, default='fps', help='fps or random')


parser.add_argument('--RL_pretrained_id', type=int, default=0, help='trial id')
parser.add_argument('--RL_ckpt_dir', type=str)
parser.add_argument('--RL_load_epoch', type=str, default='best')
parser.add_argument('--num_processes', type=int, default=1)
parser.add_argument('--save_td3_epoch', type=int, default=10)
parser.add_argument('--action_type', type=str, default='xxx')

parser.add_argument('--task_succ_margin', type=float, default=0.10)


args = parser.parse_args()

def check_shape_valid(shape_id):

    exclusive_cabinet_shape_id_list = ['41452', '45166', '45323', '45503', '45642', '45690', '45855', '45910', '45915', '45922', '46019', '46044', '46092', '46107', '46108', '46127', '46132', '46166', '46180', '46403', '46408', '46417', '46430', '46439', '46481', '46699', '46768', '46922', '46944', '47185', '47254', '47388', '47391', '47419', '47585', '47601', '47632', '47686', '47729', '47954', '47963', '48036', '48063', '48167', '48479', '48686', '48700', '48878']
    exclusive_microwave_shape_id_list = ['7306', '7273', '7292']
    if shape_id in exclusive_cabinet_shape_id_list:
        return False
    if shape_id in exclusive_microwave_shape_id_list:
        return False
    return True


def get_critic_score(input_pcs, up, forward, contact_point, waypoints, task_degree):      # 这里的pc/up/forward/ctpt默认是world坐标系的
    input_pcid1 = torch.arange(1).unsqueeze(1).repeat(1, args.num_point_per_shape).long().reshape(-1)  # BN
    if args.sample_type == 'fps':
        input_pcid2 = furthest_point_sample(input_pcs, args.num_point_per_shape).long().reshape(-1)  # BN
    elif args.sample_type == 'random':
        pcs_id = ()
        for batch_idx in range(input_pcs.shape[0]):
            idx = np.arange(input_pcs[batch_idx].shape[0])
            np.random.shuffle(idx)
            while len(idx) < args.num_point_per_shape:
                idx = np.concatenate([idx, idx])
            idx = idx[:args.num_point_per_shape]
            pcs_id = pcs_id + (torch.tensor(np.array(idx)), )
        input_pcid2 = torch.stack(pcs_id, dim=0).long().reshape(-1)
    input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(1, args.num_point_per_shape, -1)  # 1 * N * 3

    up = torch.from_numpy(up).float().unsqueeze(0).to(device)  # B x 3 # up作为feature
    forward = torch.from_numpy(forward).float().unsqueeze(0).to(device)  # B x 3   # forward
    contact_point = torch.Tensor(contact_point).unsqueeze(0).to(device)

    # get waypoints
    task_waypoints = np.array(waypoints)[:, :6]
    while len(task_waypoints) < num_steps:
        task_waypoints = np.concatenate([task_waypoints, [task_waypoints[-1]]])
    task_waypoints = torch.tensor(task_waypoints).unsqueeze(0).float().to(device)
    task_traj = torch.cat([torch.cat([up, forward], dim=1).view(1, 1, 6), task_waypoints], dim=1).view(1, (args.num_steps + 1) * 6)  # up和forward两个方向拼起来 + waypoints

    with torch.no_grad():
        pred_result_logits, pred_whole_feats = network(input_pcs, torch.tensor(task_degree, dtype=torch.float32).unsqueeze(0).to(device), task_traj, contact_point)  # B x 2, B x F x N
        critic_score = torch.sigmoid(pred_result_logits[0]).item()
    return critic_score


def run_an_RL(idx_process, args, transition_Q, epoch_Q, decay_Q):

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
    task_succ_margin = args.task_succ_margin
    if not torch.cuda.is_available():
        device = "cpu"

    # if args.random_seed is not None:
    np.random.seed(random.randint(1, 1000) + idx_process)
    random.seed(random.randint(1, 1000) + idx_process)

    train_shape_list, val_shape_list = [], []
    train_file_dir = "../stats/train_VAT_train_data_list.txt"
    val_file_dir = "../stats/train_VAT_test_data_list.txt"
    all_shape_list = []
    all_cat_list = ['StorageFurniture', 'Microwave', 'Refrigerator', 'Door']
    eval_cat_list = ['safe', 'WashingMachine']
    tot_cat = len(all_cat_list)
    len_shape = {}
    len_train_shape = {}
    shape_cat_dict = {}
    cat_shape_id_list = {}
    val_cat_shape_id_list = {}
    train_cat_shape_id_list = {}
    for cat in all_cat_list:
        len_shape[cat] = 0
        len_train_shape[cat] = 0
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
            cat_shape_id_list[cat].append(shape_id)
            val_cat_shape_id_list[cat].append(shape_id)

    EP_MAX = 1000000
    hidden_dim = 512
    policy_target_update_interval = 3 # delayed update for the policy network and target networks
    DETERMINISTIC = True  # DDPG: deterministic policy gradient
    explore_noise_scale = args.explore_noise_scale
    eval_noise_scale = args.eval_noise_scale
    if args.wp_rot:
        explore_noise_scale = torch.tensor([explore_noise_scale, explore_noise_scale, explore_noise_scale, explore_noise_scale * 5, explore_noise_scale * 5, explore_noise_scale * 5])
        eval_noise_scale = torch.tensor([eval_noise_scale, eval_noise_scale, eval_noise_scale, eval_noise_scale * 5, eval_noise_scale * 5, eval_noise_scale * 5])
    noise_decay = args.noise_decay
    threshold_gripper_distance = args.threshold

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
    if args.state_axes_all:
        state_dim += 3
    if args.state_door_dir:
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

    replay_buffer_size = args.replay_buffer_size
    replay_buffer = ReplayBuffer(replay_buffer_size)
    td3 = TD3(replay_buffer, state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, policy_target_update_interval=policy_target_update_interval, action_range=action_range, q_lr=args.q_lr, policy_lr=args.policy_lr, device=device, pred_world_xyz=args.pred_world_xyz, wp_rot=args.wp_rot).to(device)
    td3.load_model(path=os.path.join(RL_ckpt_dir, 'td3_%s' % args.RL_load_epoch))
    td3.train()

    trial_id = args.trial_id
    primact_type = args.primact_type

    out_dir = os.path.join(args.out_dir,
                           'CUR_%s_%d' % (args.action_type, trial_id))


    joint_angles = None

    # setup env
    print("creating env")
    env = Env(show_gui=(not args.no_gui))
    print("env creared")

    ### viz the EE gripper position
    # setup robot
    robot_urdf_fn = './robots/panda_gripper.urdf'
    robot_material = env.get_material(4, 4, 0.01)

    if args.state_degree:
        args.guidance_reward *= (np.pi / 180)

    if args.pred_world_xyz + args.pred_residual_world_xyz + args.pred_residual_root_qpos != 1:
        raise ValueError

    robot_loaded = 0
    tot_done_epoch = 0
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
            if args.continue_to_play:
                td3.load_model(path=os.path.join(out_dir, 'td3_%d' % ((now_epoch_qsize - 1) * save_td3_epoch + int(args.critic_load_epoch))))
            else:
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
        selected_cat = all_cat_list[random.randint(0, len(all_cat_list) - 1)]

        if args.all_shape:  # False
            shape_id = cat_shape_id_list[selected_cat][random.randint(0, len_shape[selected_cat] - 1)]
            while check_shape_valid(shape_id) is False:
                shape_id = cat_shape_id_list[selected_cat][random.randint(0, len_shape[selected_cat] - 1)]
        else:
            shape_id = train_cat_shape_id_list[selected_cat][random.randint(0, len_train_shape[selected_cat] - 1)]
            while check_shape_valid(shape_id) is False:
                shape_id = train_cat_shape_id_list[selected_cat][random.randint(0, len_train_shape[selected_cat] - 1)]

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
        task_degree = random.random() * (task_upper - task_lower) + task_lower

        if args.primact_type == 'pulling':
                joint_angles = env.update_joint_angle(joint_angles, target_part_joint_idx, target_part_state,
                                                      task_degree, push=False, pull=True)
        else:
            if args.open_door:
                joint_angles = env.update_joint_angle(joint_angles, target_part_joint_idx, target_part_state, task_degree, push=False, pull=True)
            else:
                joint_angles = env.update_joint_angle(joint_angles, target_part_joint_idx, target_part_state, task_degree, push=True, pull=False)
        env.set_object_joint_angles(joint_angles)
        env.render()
        if args.open_door:
            task_degree = - task_degree
        if args.primact_type == 'pulling':
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

        # grids
        grid_x, grid_y = np.meshgrid(np.arange(448), np.arange(448))
        grid_xy = np.stack([grid_y, grid_x]).astype(np.int32)  # 2 x 448 x 448
        # get pc
        out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
        out3 = (gt_movable_link_mask > 0).astype(np.uint8) * 255
        out3 = (np.array(out3, dtype=np.float32) > 127)
        pt = out[int(x), int(y), :3]  # contact point
        ptid = np.array([x, y], dtype=np.int32)
        mask = (out[:, :, 3] > 0.5)
        mask[x, y] = False
        pc = out[mask, :3]
        pcids = grid_xy[:, mask].T
        out3 = out3[mask]
        idx = np.arange(pc.shape[0])
        np.random.shuffle(idx)
        while len(idx) < 30000:
            idx = np.concatenate([idx, idx])
        idx = idx[:30000 - 1]
        pc = pc[idx, :]
        pcids = pcids[idx, :]
        out3 = out3[idx]
        pc = np.vstack([pt, pc])
        pcids = np.vstack([ptid, pcids])
        out3 = np.append(True, out3)
        # normalize to zero-centered
        pc[:, 0] -= 5  # normalize, x-=5
        out = torch.from_numpy(pc).unsqueeze(0).to(device)
        input_pcs = out.view(1, -1, 3).contiguous()  # B x 3N x 3   # point cloud    # type: torch.tensor
        pc_pxids = out2 = torch.from_numpy(pcids).float().unsqueeze(0).to(device)
        pc_movables = out3 = torch.from_numpy(out3).float().unsqueeze(0).to(device)

        # transfer to world coordinate system
        bs = input_pcs.shape[0]
        mat44 = torch.tensor(cam.get_metadata()['mat44'], dtype=torch.float32).to(device)
        input_pcs = (mat44[:3, :3] @ input_pcs.reshape(-1, 3).T).T.reshape(bs, -1, 3)


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
        state_door_dir = position_world - state_joint_origins

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
        end_rotmat[:3, 3] = position_world - direction_world * 0.08

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
        out_info['out_dir'] = os.path.join(out_dir, 'waypoints')
        out_info['shape_id'] = shape_id
        out_info['category'] = selected_cat
        out_info['cnt_id'] = args.cnt_id    # load_data_fromMemory会用到
        out_info['primact_type'] = args.primact_type
        out_info['trial_id'] = args.trial_id
        out_info['epoch'] = epoch
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
        out_info['input_pcs'] = input_pcs.cpu().detach().numpy().tolist()
        out_info['pc_pxids'] = pc_pxids.cpu().detach().numpy().tolist()
        out_info['pc_movables'] = pc_movables.cpu().detach().numpy().tolist()

        out_info['camera_metadata'] = cam.get_metadata_json()

        out_info['object_state'] = target_part_state
        out_info['joint_angles'] = joint_angles
        out_info['joint_angles_lower'] = env.joint_angles_lower
        out_info['joint_angles_upper'] = env.joint_angles_upper

        # move back
        robot.robot.set_root_pose(start_pose)
        robot.close_gripper()
        env.render()

        # activate contact checking
        env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)

        if args.primact_type == 'pulling':
            init_success = True
            success_grasp = False
            try:
                robot.open_gripper()
                robot.move_to_target_pose(end_rotmat, 3000)
                robot.wait_n_steps(2000)
                robot.close_gripper()
                robot.wait_n_steps(800)
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

        if args.primact_type == 'pushing':
            start_target_part_qpos = env.get_target_part_qpos()
            try:
                robot.move_to_target_pose(end_rotmat, 1500)
            except Exception:
                env.scene.remove_articulation(env.object)
                env.scene.remove_articulation(robot.robot)
                flog.close()
                continue
            robot.wait_n_steps(1500)
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
        out_info['out_of_threshold'] = 0
        waypoints = []

        success = True
        success_traj = False
        transition_list = []

        initial_gripper_finger_position = position_world - 0.02 * action_direction_world

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
            if args.state_door_dir:
                state_RL.extend(state_door_dir.tolist())
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
                    if args.primact_type == 'pushing':
                        imgs = robot.move_to_target_pose(final_rotmat, 2500, cam=cam, vis_gif=True, vis_gif_interval=500)
                        succ_images.extend(imgs)
                        robot.wait_n_steps(600)
                    if args.primact_type == 'pulling':
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

                now_state_door_dir = now_contact_point_position - state_joint_origins

                now_gripper_root_position = position_world - 0.15 * action_direction_world + now_qpos[0] * forward + now_qpos[1] * left + now_qpos[2] * up

                r = R.from_euler('XYZ', [now_qpos[3], now_qpos[4], now_qpos[5]], degrees=False)
                now_rotmat = final_rotmat[:3, :3] @ r.as_matrix()
                now_up = now_rotmat[:3, 2]
                now_gripper_finger_position = now_gripper_root_position + now_up * 0.13

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
                if args.state_door_dir:
                    state_.extend(now_state_door_dir.tolist())
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
                reward -= 150 * np.linalg.norm(now_contact_point_position - now_gripper_finger_position)
                tot_penalty -= 150 * np.linalg.norm(now_contact_point_position - now_gripper_finger_position)
                if np.linalg.norm(now_contact_point_position - now_gripper_finger_position) > threshold_gripper_distance:
                    reward -= 300
                    tot_penalty -= 300
                if np.linalg.norm(now_contact_point_position - now_gripper_finger_position) > 0.4:
                    out_info['out_of_threshold'] = 1

                if distance < degree2radian(abs(task_degree) * task_succ_margin):  # bonus
                    reward += args.success_reward
                    bonus += args.success_reward
                    stop = True
                    end_step = step_idx + 1
                    tot_done_epoch += 1
                    # export SUCC GIF Image
                    try:
                        imageio.mimsave(os.path.join(result_succ_dir, '%d_%d_%.3f_%.3f_%.3f_%d_%s.gif' % (tot_done_epoch, idx_process, task_degree, radian2degree(init_target_part_qpos), radian2degree(joint_angle_upper), step_idx+1, selected_cat)), succ_images)
                    except:
                        pass

                elif step_idx == num_steps - 1:
                    stop = True
                    end_step = step_idx + 1
                    if tot_fail_epoch < tot_done_epoch:
                        tot_fail_epoch += 1
                        try:
                            imageio.mimsave(os.path.join(result_fail_dir, '%d_%d_%.3f_%.3f_%.3f_%.3f_%d_%s.gif' % (tot_fail_epoch, idx_process, task_degree, radian2degree(init_target_part_qpos), radian2degree(joint_angle_upper), radian2degree(distance), step_idx+1, selected_cat)), succ_images)
                        except:
                            pass

                # store (s,a,r)
                if stop and distance < degree2radian(abs(task_degree) * task_succ_margin):
                    success_traj = True
                replay_buffer.push(cur_state.view(-1).cpu().detach().numpy(), next_qpos.view(-1).cpu().detach().numpy(), reward,
                                    next_state.view(-1).cpu().detach().numpy(), done=stop)

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
        out_info['actual_task'] = radian2degree(init_target_part_qpos - env.get_target_part_qpos())

        if args.no_gui:
            # close env
            # env.close()
            pass
        else:
            if success:
                print('[Successful Interaction] Done. Ctrl-C to quit.')
                ### wait forever
                # robot.wait_n_steps(100000000000)
                # env.close()
            else:
                print('[Unsuccessful Interaction] invalid gripper-object contact.')
                # close env
                # env.close()

        final_reward = reward
        traj_info = [final_distance, ep_r, final_reward, bonus, tot_penalty, tot_guidance]
        cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]
        transition_Q.put([idx_process, transition_list, waypoints, success_traj, traj_info, out_info, cam_XYZA_list, gt_all_movable_link_mask, gt_movable_link_mask])

        flog.close()
        env.scene.remove_articulation(env.object)

trial_id = args.trial_id
primact_type = args.primact_type

out_dir = os.path.join(args.out_dir,
                       'CUR_%s_%d' % (args.action_type, trial_id))
RL_ckpt_dir = args.RL_ckpt_dir
if os.path.exists(out_dir):
    response = input('Out directory "%s" already exists, overwrite? (y/n) ' % out_dir)
    if response != 'y' and response != 'Y':
        sys.exit()
    shutil.rmtree(out_dir)
os.makedirs(out_dir)
print('out_dir: ', out_dir)
os.makedirs(os.path.join(out_dir, 'critic_ckpts'))
result_succ_dir = os.path.join(out_dir, 'result_succ_imgs')
if not os.path.exists(result_succ_dir):
    os.mkdir(result_succ_dir)
result_fail_dir = os.path.join(out_dir, 'result_fail_imgs')
if not os.path.exists(result_fail_dir):
    os.mkdir(result_fail_dir)

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
if args.state_axes_all:
    state_dim += 3
if args.state_door_dir:
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


# load critic
critic_replay_buffer = utils.CriticReplayBuffer(args.critic_replay_buffer_size)

critic_exp_name = f'exp-{args.critic_model_version}-{args.action_type}-{args.critic_exp_suffix}'
critic_exp_dir = os.path.join(args.critic_log_dir, critic_exp_name)
model_def = utils.get_model_module(args.critic_model_version)
network = model_def.Network(args.critic_feat_dim, num_steps=args.num_steps + 1).to(device)
print('critic_exp_dir', critic_exp_dir)
if args.continue_to_play:
    network.load_state_dict(torch.load(os.path.join(args.critic_exp_dir, 'critic_ckpts', '%s-network.pth' % args.critic_load_epoch)))
else:
    network.load_state_dict(torch.load(os.path.join(critic_exp_dir, 'ckpts', '%s-network.pth' % args.critic_load_epoch)))
# create optimizers
network_opt = torch.optim.Adam(network.parameters(), lr=args.critic_lr, weight_decay=args.critic_weight_decay)
# learning rate scheduler
network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=args.critic_lr_decay_every, gamma=args.critic_lr_decay_by)
# send parameters to device
utils.optimizer_to_device(network_opt, device)


# RL
replay_buffer_size = args.replay_buffer_size
replay_buffer = ReplayBuffer(replay_buffer_size)
td3 = TD3(replay_buffer, state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, policy_target_update_interval=policy_target_update_interval, action_range=action_range, q_lr=args.q_lr, policy_lr=args.policy_lr, device=device, pred_world_xyz=args.pred_world_xyz, wp_rot=args.wp_rot).to(device)
td3.load_model(path=os.path.join(RL_ckpt_dir, 'td3_%s' % args.RL_load_epoch))
td3.train()

# load dataset
data_features = ['pcs', 'pc_pxids', 'pc_movables', 'gripper_img_target', 'gripper_direction',
                 'gripper_forward_direction', \
                 'result', 'task_motion', 'gt_motion', 'task_waypoints', 'cur_dir', 'shape_id', 'trial_id',
                 'is_original', 'position']

# config for critic
conf = ArgumentParser()
conf.device = device
conf.num_point_per_shape = args.num_point_per_shape
conf.batch_size = args.critic_batch_size
conf.num_steps = args.num_steps + 1
conf.no_visu = True
conf.critic_score_threshold = args.critic_score_threshold
conf.sample_type = args.sample_type
critic_step = 0
critic_num = 0
critic_true_num = 0

epoch = -1
if args.continue_to_play:
    epoch = int(args.RL_load_epoch) - 1
record = [0 for idx in range(100)]
num_type1, num_type2, num_type3 = 0, 0, 0
# unsuccess, success + high critic_score, success + low critic_score
tb_writer = SummaryWriter(os.path.join(out_dir, 'tb_logs'))
best_acc = 0.0
accuracy = 0.0
task_succ_margin = args.task_succ_margin

t0 = time.time()
while True:
    if not trans_q.empty():
        epoch += 1
        if epoch % args.save_td3_epoch == 0:
            td3.save_model(path=os.path.join(out_dir, 'td3_%d_%d' % (epoch, int(accuracy * 100))))
            td3.save_model(path=os.path.join(out_dir, 'td3_%d' % (epoch)))
            epoch_q.put(epoch)
        if epoch % args.save_critic_epoch == 0:
            # save critic ckpt
            torch.save(network.state_dict(), os.path.join(out_dir, 'critic_ckpts', '%d-network.pth' % epoch))
            torch.save(network_opt.state_dict(), os.path.join(out_dir, 'critic_ckpts', '%d-optimizer.pth' % epoch))
            torch.save(network_lr_scheduler.state_dict(), os.path.join(out_dir, 'critic_ckpts', '%d-lr_scheduler.pth' % epoch))
        if epoch % decay_interval == 0 and epoch != 0:
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
            record[epoch % 100] = 1
        else:
            record[epoch % 100] = 0

        # action_direction_world, forward, position_world, task_degree, x, y = traj_setting
        action_direction_world = np.array(out_info['gripper_direction_world'])
        forward = np.array(out_info['gripper_forward_direction_world'])
        position_world = np.array(out_info['position_world'])
        task_degree = out_info['task']
        x, y = out_info['pixel_locs']
        input_pcs = torch.tensor(np.array(out_info['input_pcs'], dtype=np.float32)).to(device)
        for step_idx in range(len_episode):
            sar = transition_list[step_idx]
            cur_state, next_qpos, reward, next_state, stop = sar
            if success_traj and (step_idx == len_episode - 1):
                real_critic_score = get_critic_score(input_pcs, action_direction_world, forward, position_world, waypoints, task_degree)
                reward -= args.lbd_critic_penalty * real_critic_score
                ep_r -= args.lbd_critic_penalty * real_critic_score
                print('critic_score: ', real_critic_score)
                tb_writer.add_scalar('critic_score', real_critic_score, critic_num)
                tb_writer.add_scalar('critic_score_epoch', epoch, critic_num)
                if real_critic_score >= args.critic_score_threshold:
                    critic_true_num += 1
                else:
                    critic_num += 1
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
                    done = False

                    # guidance reward
                    reward = args.guidance_reward * (np.abs(cur_state[1]) - np.abs(next_state[1]))  # 10 degree, reward ~ 0.87
                    guidance = args.guidance_reward * (np.abs(cur_state[1]) - np.abs(next_state[1]))

                    reward -= 150 * np.linalg.norm(now_contact_point_position - now_gripper_position)
                    if np.linalg.norm(now_contact_point_position - now_gripper_position) > threshold_gripper_distance:
                        reward -= 300
                        if args.HER_only_attach:
                            continue

                    if abs(next_position - new_goal) < task_succ_margin * abs(degree2radian(task_degree)):
                        # bonus
                        reward += args.success_reward
                        done = True

                        new_task_degree = radian2degree(new_goal - init_target_part_qpos)
                        fake_critic_score = get_critic_score(input_pcs, action_direction_world, forward, position_world, waypoints[:i+1], new_task_degree)
                        reward -= args.lbd_critic_penalty * fake_critic_score
                        # reward -= guidance * fake_critic_score

                    new_task = np.array([new_goal - init_target_part_qpos])
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


        critic_update = False
        if success_traj and (real_critic_score < args.critic_score_threshold) and out_info['out_of_threshold'] == 0:
            critic_update = True
            num_type3 += 1
            out_info['type'] = 3

            print('save this epoch')
            with open(os.path.join(out_dir, 'result_%d.json' % epoch), 'w') as fout:
                json.dump(out_info, fout)
            critic_replay_buffer.push(out_info)

            save_h5(os.path.join(out_dir, 'cam_XYZA_%d.h5' % epoch), [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                                                                      (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                                                                      (cam_XYZA_pts.astype(np.float32), 'pc',
                                                                       'float32')])
            Image.fromarray((gt_movable_link_mask > 0).astype(np.uint8) * 255).save(
                os.path.join(out_dir, 'interaction_mask_%d.png' % epoch))

        if success_traj and (real_critic_score >= args.critic_score_threshold) and (num_type2 < num_type3) and out_info['out_of_threshold'] == 0:
            out_info['type'] = 2
            num_type2 += 1
            critic_replay_buffer.push(out_info)

        if (not success_traj) and (num_type1 < num_type3) and (final_distance > args.critic_degree_lower) and out_info['out_of_threshold'] == 0:
            out_info['type'] = 1
            num_type1 += 1
            critic_replay_buffer.push(out_info)

        if args.critic_update_frequently or critic_update:
            print('len_critic_buffer:', len(critic_replay_buffer))
            if len(critic_replay_buffer) >= args.critic_batch_size:
                for critic_update_idx in range(args.critic_update_itr):
                    critic_step += 1
                    out_info_batch = critic_replay_buffer.sample(args.critic_batch_size)
                    train_dataset = SAPIENVisionDataset([args.primact_type], args.category, data_features,
                                                        args.critic_replay_buffer_size, img_size=args.img_size,
                                                        no_true_false_equal=False, angle_system=1,
                                                        degree_lower=args.critic_degree_lower,
                                                        degree_upper=args.critic_degree_upper,
                                                        critic_mode=True)
                    train_dataset.load_data_fromMemory(out_info_batch, coordinate_system=args.coordinate_system)
                    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.critic_batch_size,
                                                                   shuffle=True,
                                                                   pin_memory=True, num_workers=0, drop_last=True,
                                                                   collate_fn=utils.collate_feats,
                                                                   worker_init_fn=utils.worker_init_fn)
                    train_batches = enumerate(train_dataloader, 0)

                    total_loss = 0
                    train_batch_num = 0
                    for train_batch_ind, batch in train_batches:
                        # set models to training mode
                        network.train()
                        # forward pass (including logging)
                        loss, whole_feats, whole_pcs = critic_forward(batch=batch,
                                                                       data_features=data_features,
                                                                       network=network,
                                                                       conf=conf,
                                                                       is_val=False,
                                                                       step=critic_step,
                                                                       epoch=epoch,
                                                                       batch_ind=train_batch_ind,
                                                                       num_batch=1,
                                                                       start_time=time.time(),
                                                                       log_console=None,
                                                                       log_tb=True,
                                                                       tb_writer=tb_writer,
                                                                       lr=network_opt.param_groups[0]['lr'])
                        total_loss += loss.item()
                        train_batch_num += 1

                        # optimize one step
                        network_opt.zero_grad()
                        loss.backward()
                        network_opt.step()
                        network_lr_scheduler.step()
                    print('critic_loss: ', total_loss / train_batch_num)

        accuracy = sum(record) / len(record)
        if best_acc < accuracy + 0.001:
            best_acc = accuracy
            td3.save_model(path=os.path.join(out_dir, 'td3_best'))

        tb_writer.add_scalar('accuracy', accuracy, epoch + 1)

        print(
            'Episode: {}/{}  | Final Distance: {:.4f}  | Episode Reward: {:.4f}  | Episode Final Reward: {:.4f}  | Bonus: {:.4f}  | Penalty: {:.4f}  | Accuracy: {:.4f} | Running Time: {:.4f}'.format(
                epoch, EP_MAX, final_distance, ep_r, final_reward, bonus, tot_penalty, accuracy,
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



