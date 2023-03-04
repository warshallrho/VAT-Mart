import os
import sys
import numpy as np
from PIL import Image
from utils import get_global_position_from_camera, save_h5, radian2degree, degree2radian
import json
from argparse import ArgumentParser
import torch
import time

from sapien.core import Pose, ArticulationJointType
from env import Env, ContactError
from camera import Camera
from robots.panda_robot import Robot
import random
from scipy.spatial.transform import Rotation as R
from models.model_3d_task_actor_iclr import ActorNetwork
from models.model_3d_task_score_topk import ActionScore
from models.model_3d_task_critic_updir_RL import Network as Critic
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import torch.nn.functional as F
from data_iclr import SAPIENVisionDataset
import utils
import imageio

import multiprocessing as mp
# import render_using_blender as render_utils

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--category', type=str)
parser.add_argument('--primact_type', type=str)
parser.add_argument('--novel_cat', action='store_true', default=False)

parser.add_argument('--out_dir', type=str)
parser.add_argument('--trial_id', type=str, default='xxx', help='trial id')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--feat_dim', type=int, default=128)
parser.add_argument('--num_steps', type=int, default=5)
parser.add_argument('--out_gif', type=int, default=0)
parser.add_argument('--wp_rot', action='store_true', default=False, help='no_gui [default: False]')

parser.add_argument('--num_point_per_shape', type=int, default=10000)
parser.add_argument('--wp_xyz', type=int, default=1)
parser.add_argument('--coordinate_system', type=str, default='cambase')
parser.add_argument('--sample_type', type=str, default='random')
parser.add_argument('--affordance_dir', type=str, default='xxx')
parser.add_argument('--affordance_epoch', type=int, default=0)
parser.add_argument('--critic_dir', type=str, default='xxx')
parser.add_argument('--critic_epoch', type=str, default='0')
parser.add_argument('--actor_dir', type=str, default='xxx')
parser.add_argument('--actor_epoch', type=str, default='0')
parser.add_argument('--val_data_dir', type=str, help='data directory')
parser.add_argument('--val_data_dir2', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir3', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir4', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir5', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir6', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir7', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir8', type=str, default='xxx', help='data directory')
parser.add_argument('--val_num_data_uplimit', type=int, default=100000)
parser.add_argument('--angle_system', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='save_dir', help='data directory')
parser.add_argument('--succ_thresh', type=float, default=0.15)
parser.add_argument('--num_processes', type=int, default=2)
parser.add_argument('--num_offset', type=int, default=0)
parser.add_argument('--visu_aff', action='store_true', default=False)




args = parser.parse_args()
ctx = torch.multiprocessing.get_context("spawn")

def bgs(d6s):
    # print(d6s[0, 0, 0] *d6s[0, 0, 0] + d6s[0, 1, 0] * d6s[0, 1, 0] + d6s[0, 2, 0] *d6s[0, 2, 0])
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    # print(torch.stack([b1, b2, b3], dim=1).shape)
    # print(torch.stack([b1, b2, b3], dim=1)[0])
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

def append_data_list(file_dir, data_list, only_true_data=False):
    if file_dir != 'xxx':
        for root, dirs, files in os.walk(file_dir):
            for dir in dirs:
                data_list.append(os.path.join(file_dir, dir))
            break
    return data_list


# load data
#################
val_data_list = []
val_data_list = append_data_list(args.val_data_dir, val_data_list)
val_data_list = append_data_list(args.val_data_dir2, val_data_list)
val_data_list = append_data_list(args.val_data_dir3, val_data_list)
val_data_list = append_data_list(args.val_data_dir4, val_data_list)
val_data_list = append_data_list(args.val_data_dir5, val_data_list)
val_data_list = append_data_list(args.val_data_dir6, val_data_list)
val_data_list = append_data_list(args.val_data_dir7, val_data_list)
val_data_list = append_data_list(args.val_data_dir8, val_data_list)

data_features = ['pcs', 'gt_motion', 'shape_id', 'camera_metadata', 'joint_angles', 'ori_pixel_ids', 'epoch_idx']

device = args.device
if not torch.cuda.is_available():
    device = "cpu"


def worker(idx_process, args, transition_Q):
    # set random seed
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    else:
        np.random.seed(random.randint(1, 1000) + idx_process)
        random.seed(random.randint(1, 1000) + idx_process)

    batch_size = 1
    primact_type = args.primact_type

    # load actor
    actor = ActorNetwork(feat_dim=args.feat_dim, num_steps=5).to(device)
    affordance = ActionScore(feat_dim=args.feat_dim).to(device)
    critic = Critic(feat_dim=args.feat_dim, num_steps=5).to(device)

    actor.load_state_dict(torch.load(os.path.join(args.actor_dir, 'ckpts', '%s-network.pth' % (args.actor_epoch))))
    affordance.load_state_dict(torch.load(os.path.join(args.affordance_dir, 'ckpts', '%s-network.pth' % (args.affordance_epoch))))
    critic.load_state_dict(torch.load(os.path.join(args.critic_dir, 'ckpts', '%s-network.pth' % (args.critic_epoch))))

    actor.eval()
    affordance.eval()
    critic.eval()


    # setup env
    print("creating env")
    env = Env(show_gui=(not args.no_gui))
    print("env creared")
    cam = Camera(env, random_position=True, restrict_dir=True)
    print("camera created")
    if not args.no_gui:
        env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)


    # setup robot
    robot_urdf_fn = './robots/panda_gripper.urdf'
    robot_material = env.get_material(4, 4, 0.01)
    robot_loaded = 0

    object_material = env.get_material(4, 4, 0.01)

    lower_bound = idx_process * batch_num_per_thread
    upper_bound = min((idx_process + 1) * batch_num_per_thread, val_num_batch)
    print('process %d: lower_bound: %d, upper_bound: %d' % (idx_process, lower_bound, upper_bound))
    cur_data_batches = val_batches[lower_bound: upper_bound]

    for batch_id, batch in cur_data_batches:
        torch.cuda.empty_cache()

        camera_metadata = batch[data_features.index('camera_metadata')][0]
        cam_theta, cam_phi = camera_metadata['theta'], camera_metadata['phi']
        cam.change_pose(phi=cam_phi, theta=cam_theta, random_position=False, restrict_dir=True)
        if not args.no_gui:
            env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)

        epoch_idx = batch[data_features.index('epoch_idx')][0]
        shape_id = batch[data_features.index('shape_id')][0]
        print('shape_id: ', shape_id)
        object_urdf_fn = '../data/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % shape_id
        # target_part_state = batch[data_features.index('object_state')][0]
        target_part_state = 'random-middle'
        joint_angles = batch[data_features.index('joint_angles')][0].tolist()
        _ = env.load_object(object_urdf_fn, object_material, state=target_part_state, target_part_id=-1)
        env.set_object_joint_angles(joint_angles)
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
            continue


        # get movable link mask
        rgb, depth = cam.get_observation()
        object_link_ids = env.movable_link_ids
        gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)   # all movable link mask

        # rgb_pose, _ = cam.get_observation()
        # fimg = (rgb_pose * 255).astype(np.uint8)
        # Image.fromarray(fimg).save(
        #     os.path.join(save_dir, 'fimg_%d_%s.png' % (batch_id, shape_id)))

        # sample a pixel to interact
        xs, ys = np.where(gt_movable_link_mask > 0)
        if len(xs) == 0:
            env.scene.remove_articulation(env.object)
            continue
        pixel_locs = batch[data_features.index('ori_pixel_ids')][0]
        x, y = pixel_locs[0], pixel_locs[1]
        target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
        env.set_target_object_part_actor_id(target_part_id)
        gt_movable_link_mask = cam.get_movable_link_mask([target_part_id])  # target movable link mask

        task_degree = batch[data_features.index('gt_motion')][0]
        task = degree2radian(task_degree)
        print("task:", task_degree)


        ### use the GT vision
        rgb, depth = cam.get_observation()
        cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
        cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])

        out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
        out3 = (gt_movable_link_mask > 0).astype(np.uint8) * 255
        out3 = np.array(out3, dtype=np.float32) > 127
        pt = out[x, y, :3]
        mask = (out[:, :, 3] > 0.5)
        mask[x, y] = False
        pc = out[mask, :3]
        out3 = out3[mask]
        idx = np.arange(pc.shape[0])
        np.random.shuffle(idx)
        while len(idx) < 30000:
            idx = np.concatenate([idx, idx])
        idx = idx[:30000 - 1]
        out3 = out3[idx]
        pc = pc[idx, :]
        pc = np.vstack([pt, pc])
        out3 = np.append(True, out3)
        input_movables = out3
        pc[:, 0] -= 5   # cam, norm

        pc = (cam.get_metadata()['mat44'][:3, :3] @ pc.T).T

        input_pcs = torch.tensor(pc, dtype=torch.float32).reshape(1, 30000, 3).float().contiguous().to(device)
        input_movables = torch.tensor(input_movables, dtype=torch.float32).reshape(1, 30000, 1).to(device)

        input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, args.num_point_per_shape).long().reshape(-1)  # BN
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
        world_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, args.num_point_per_shape, -1)
        input_movables = input_movables[input_pcid1, input_pcid2, :].reshape(batch_size, args.num_point_per_shape, -1)

        task_degree_tensor = torch.from_numpy(np.array(task_degree)).float().view(1, 1).to(device)

        with torch.no_grad():
            pred_action_score_map = affordance.inference_action_score(world_pcs, task_degree_tensor).cpu().numpy()
        pred_action_score_map = pred_action_score_map * input_movables.cpu().numpy()

        # render aff_map
        if args.visu_aff:
            fn = os.path.join(save_dir, 'affordance', 'pred_%d_%s' % (epoch_idx, shape_id))
            pc_camera = (np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ world_pcs[0].detach().cpu().numpy().T).T
            utils.render_pts_label_png(fn, pc_camera, pred_action_score_map[0])

        aff_max_idx = np.argmax(pred_action_score_map)
        aff_pos = world_pcs.view(-1, 3)[aff_max_idx].view(1, 3)
        # print("aff:", aff_pos)
        position_world = aff_pos.reshape(3).detach().cpu().numpy()

        with torch.no_grad():
            traj = actor.sample_n(world_pcs, task_degree_tensor, aff_pos, rvs=100)
            scores = torch.sigmoid(critic.forward_n(world_pcs, task_degree_tensor, traj, aff_pos, rvs=100)[0])

        top_score, top_idx = scores.view(1, 100, 1).max(dim=1)
        recon_traj = traj[top_idx][0]
        recon_dir = recon_traj[:, 0, :]
        recon_dir = recon_dir.reshape(-1, 2, 3).permute(0, 2, 1)
        recon_dir = bgs(recon_dir).detach().cpu().numpy()
        recon_wps = recon_traj[:, 1:, :].detach().cpu().numpy()

        up = recon_dir[0, :, 0]
        forward = recon_dir[0, :, 1]
        left = recon_dir[0, :, 2]

        rotmat = np.eye(4).astype(np.float32)
        rotmat[:3, 0] = forward
        rotmat[:3, 1] = left
        rotmat[:3, 2] = up
        action_direction_world = up

        start_rotmat = np.array(rotmat, dtype=np.float32)
        start_rotmat[:3, 3] = position_world - action_direction_world * 0.15  # add displacement(lase column)
        start_pose = Pose().from_transformation_matrix(start_rotmat)

        end_rotmat = start_rotmat.copy()
        end_rotmat[:3, 3] = position_world - action_direction_world * 0.08


        out_info = dict()
        out_info['out_dir'] = out_dir
        out_info['shape_id'] = shape_id
        out_info['category'] = args.category
        out_info['primact_type'] = args.primact_type
        out_info['epoch_idx'] = epoch_idx
        out_info['random_seed'] = args.random_seed
        out_info['task'] = task_degree
        out_info['pixel_locs'] = [int(x), int(y)]
        out_info['target_object_part_actor_id'] = env.target_object_part_actor_id
        out_info['target_object_part_joint_id'] = env.target_object_part_joint_id
        if env.target_object_part_joint_type == ArticulationJointType.REVOLUTE:
            out_info['target_object_part_joint_type'] = "REVOLUTE"
        elif env.target_object_part_joint_type == ArticulationJointType.PRISMATIC:
            out_info['target_object_part_joint_type'] = 'PRISMATIC'
        else:
            out_info['target_object_part_joint_type'] = str(env.target_object_part_joint_type)
        out_info['camera_metadata'] = cam.get_metadata_json()
        out_info['mat44'] = str(cam.get_metadata()['mat44'])
        out_info['gripper_direction_world'] = action_direction_world.tolist()
        out_info['position_world'] = position_world.tolist()
        out_info['gripper_forward_direction_world'] = forward.tolist()
        out_info['object_state'] = target_part_state
        out_info['joint_angles'] = joint_angles
        out_info['start_rotmat_world'] = start_rotmat.tolist()



        if robot_loaded == 0:
            robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))
            robot_loaded = 1
        else:
            robot.load_gripper(robot_urdf_fn, robot_material)
        env.end_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)

        final_distance = 100

        # move back
        robot.robot.set_root_pose(start_pose)
        env.render()

        if not args.no_gui:
            env.wait_to_start()
            pass

        out_info['grasp_error'] = 0

        if args.primact_type == 'pushing':
            robot_move_steps = 2500
        elif args.primact_type == 'pulling':
            robot_move_steps = 3000

        if args.primact_type == 'pulling':
            init_success = True
            success_grasp = False
            try:
                robot.open_gripper()
                robot.move_to_target_pose(end_rotmat, robot_move_steps)
                robot.wait_n_steps(1000)
                robot.close_gripper()
                robot.wait_n_steps(600)
                now_qpos = robot.robot.get_qpos().tolist()
                finger1_qpos = now_qpos[-1]
                finger2_qpos = now_qpos[-2]
                # print(finger1_qpos, finger2_qpos)
                if finger1_qpos + finger2_qpos > 0.01:
                    success_grasp = True
            except Exception:
                init_success = False
            if not (success_grasp and init_success):
                out_info['grasp_error'] = 1
                # env.scene.remove_articulation(env.object)
                # env.scene.remove_articulation(robot.robot)
                # continue


        ### main steps
        out_info['start_target_part_qpos'] = env.get_target_part_qpos()
        init_target_part_qpos = env.get_target_part_qpos()

        num_steps = args.num_steps  # waypoints = num_step + 1
        out_info["num_steps"] = num_steps + 1
        waypoints = []
        dense_waypoints = []

        success = True
        robot.close_gripper()
        env.render()

        try:
            robot.wait_n_steps(400)
            succ_images = []
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose*255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            succ_images.append(fimg)
            begin_img = fimg

            for step_idx in range(num_steps):
                waypoint = recon_wps[0, step_idx]
                # print(waypoint)

                # get rotmat and move
                final_rotmat = start_rotmat.copy()

                final_rotmat[0, 3] += waypoint[0]
                final_rotmat[1, 3] += waypoint[1]
                final_rotmat[2, 3] += waypoint[2]

                if args.wp_rot:
                    try:
                        r = R.from_euler('XYZ', [waypoint[3], waypoint[4], waypoint[5]], degrees=False)
                        final_rotmat[:3, :3] = final_rotmat[:3, :3] @ r.as_matrix()
                    except Exception:
                        success = False
                        break

                try:
                    imgs, cur_waypoints = robot.move_to_target_pose(final_rotmat, robot_move_steps, cam=cam, vis_gif=True, vis_gif_interval=500, visu=True)
                    dense_waypoints.extend(cur_waypoints)
                    cur_waypoints = robot.wait_n_steps(1000, visu=True)
                    dense_waypoints.extend(cur_waypoints)
                    if args.primact_type == 'pulling':
                        robot.close_gripper()
                        cur_waypoints = robot.wait_n_steps(400, visu=True)
                        dense_waypoints.extend(cur_waypoints)
                except Exception:
                    success = False
                    break
                cur_waypoint = robot.robot.get_qpos().tolist()
                waypoints.append(cur_waypoint)

                if args.out_gif:
                    succ_images.extend(imgs)

                ''' calculate reward  (use radian) '''
                stop = False
                distance = np.abs((init_target_part_qpos - task) - env.get_target_part_qpos())
                final_distance = radian2degree(distance)
                # print("dis:", radian2degree((init_target_part_qpos - task) - env.get_target_part_qpos()))

                if distance < degree2radian(abs(task_degree) * args.succ_thresh):
                    stop = True
                    if args.out_gif:
                        try:
                            imageio.mimsave(os.path.join(result_succ_dir, '%d_%d_%.3f_%.3f_%d.gif' % (args.num_offset, epoch_idx, task_degree, radian2degree(init_target_part_qpos), step_idx+1)), succ_images)
                        except:
                            pass

                if step_idx == num_steps - 1 and distance >= degree2radian(abs(task_degree) * args.succ_thresh):
                    stop = True
                    if args.out_gif:
                        try:
                            imageio.mimsave(os.path.join(result_fail_dir, '%d_%d_%.3f_%.3f_%.3f_%d.gif' % (args.num_offset, epoch_idx, task_degree, radian2degree(init_target_part_qpos), radian2degree(distance), step_idx+1)), succ_images)
                        except:
                            pass

                if stop:
                    break

            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose*255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            end_img = fimg

        except ContactError:
            success = False

        out_info['final_target_part_qpos'] = env.get_target_part_qpos()
        actual_movement = radian2degree(init_target_part_qpos - env.get_target_part_qpos())
        out_info['actual_movement'] = actual_movement
        out_info['final_distance'] = final_distance

        # close the file
        env.scene.remove_articulation(robot.robot)
        env.scene.remove_articulation(env.object)

        if success:
            out_info['result'] = 'VALID'
        else:
            out_info['result'] = 'CONTACT_ERROR'
            continue

        out_info['waypoints'] = waypoints
        out_info['dense_waypoints'] = dense_waypoints

        if args.no_gui:
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


        cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]
        frames = [begin_img, end_img]
        transition_Q.put([idx_process, out_info, cam_XYZA_list, gt_movable_link_mask, success, frames])

    env.close()






# out_dir = os.path.join(args.out_dir, '%s_%s_%d_%s_%d' % (shape_id, args.category, args.cnt_id, primact_type, trial_id))
out_dir = os.path.join(args.out_dir, str(args.trial_id))
print('out_dir: ', out_dir)

save_dir = os.path.join(out_dir, args.save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# gen visu
# visu_dir = os.path.join(conf.exp_dir, 'val_visu')
# out_dir = os.path.join(visu_dir, 'epoch-%04d' % epoch)
# input_pc_dir = os.path.join(out_dir, 'input_pc')

result_succ_dir = os.path.join(save_dir, 'result_succ_imgs')
if not os.path.exists(result_succ_dir):
    os.mkdir(result_succ_dir)
result_fail_dir = os.path.join(save_dir, 'result_fail_imgs')
if not os.path.exists(result_fail_dir):
    os.mkdir(result_fail_dir)

if args.visu_aff and not os.path.exists(os.path.join(save_dir, 'affordance')):
    os.mkdir(os.path.join(save_dir, 'affordance'))

val_dataset = SAPIENVisionDataset([args.primact_type], [], data_features, buffer_max_num=512,
                                  img_size=224, only_true_data=True,
                                  no_true_false_equal=False, angle_system=args.angle_system,
                                  EP_MAX=30000, degree_lower=10,
                                  cur_primact_type=args.primact_type, critic_mode=False, train_mode=False)
val_dataset.load_data(val_data_list, wp_xyz=args.wp_xyz, coordinate_system=args.coordinate_system,
                      num_data_uplimit=args.val_num_data_uplimit)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True,
                                             num_workers=0, drop_last=True, collate_fn=utils.collate_feats,
                                             worker_init_fn=utils.worker_init_fn)
val_num_batch = len(val_dataloader)
batch_num_per_thread = val_num_batch // args.num_processes + 1
val_batches = list(enumerate(val_dataloader, 0))



def main():
    trans_q = ctx.Queue()

    for idx_process in range(args.num_processes):
        p = ctx.Process(target=worker, args=(idx_process, args, trans_q))
        p.start()


    cur_epoch = 0
    num_succ = 0
    EP_MAX = 5000
    num_contact_error, num_grasp_error = 0, 0

    t0 = time.time()
    while True:
        if not trans_q.empty():
            cur_epoch += 1

            idx_process, out_info, cam_XYZA_list, gt_movable_link_mask, success, frames = trans_q.get()
            epoch_idx = out_info['epoch_idx']
            cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA = cam_XYZA_list
            begin_img, end_img = frames

            task_degree = out_info['task']
            actual_movement = out_info['actual_movement']
            final_distance = out_info['final_distance']
            result = out_info['result']
            grasp_error = out_info['grasp_error']

            if abs(task_degree - actual_movement) <= (abs(task_degree) * args.succ_thresh):
                num_succ += 1
            # print('success_trajectory: ', num_succ)

            if result == 'CONTACT_ERROR':
                num_contact_error += 1
            if result == grasp_error:
                num_grasp_error += 1


            # save results
            with open(os.path.join(save_dir, 'result_%d_%d.json' % (args.num_offset, epoch_idx)), 'w') as fout:
                json.dump(out_info, fout)
            save_h5(os.path.join(save_dir, 'cam_XYZA_%d_%d.h5' % (args.num_offset, epoch_idx)),
                    [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                     (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                     (cam_XYZA_pts.astype(np.float32), 'pc', 'float32')])
            Image.fromarray((gt_movable_link_mask > 0).astype(np.uint8) * 255).save(
                os.path.join(save_dir, 'interaction_mask_%d_%d.png' % (args.num_offset, epoch_idx)))
            begin_img.save(os.path.join(save_dir, 'begin_%d_%d.png' % (args.num_offset, epoch_idx)))  # first frame
            end_img.save(os.path.join(save_dir, 'end_%d_%d.png' % (args.num_offset, epoch_idx)))  # first frame


            accuracy = num_succ / cur_epoch
            print(
                'Epi: {}/{}  | Dis: {:.4f}  | Task: {:.4f}  |  Rate: {:.4f}  | Accuracy: {:.4f}  | Running Time: {:.4f}'.format(
                    cur_epoch, EP_MAX, final_distance, task_degree, final_distance / task_degree, accuracy, time.time() - t0)
            )

if __name__=='__main__':
    sys.exit(main())