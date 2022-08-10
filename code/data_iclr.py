import os
import h5py
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json
from progressbar import ProgressBar
from camera import Camera
import random


class SAPIENVisionDataset(data.Dataset):

    def __init__(self, primact_types, category_types, data_features, buffer_max_num, img_size=224, \
            no_true_false_equal=False, no_neg_dir_data=False, only_true_data=False, joint_types='REVOLUTE', \
            angle_system=1, EP_MAX=2000, degree_lower=10.0, degree_upper=45.0, \
            curiosityDriven=False, cur_shape_id=48855, cur_category='StorageFurniture', cur_cnt_id=0, cur_primact_type='pushing', cur_trial_id=32504, waypoints_dim=6, critic_mode=True, train_mode=True, affordance_mode=False):
        self.primact_types = primact_types
        self.category_types = category_types
        if joint_types is None:
            self.joint_types = ['REVOLUTE', 'PRISMATIC']
        else:
            self.joint_types = [joint_types]

        self.buffer_max_num = buffer_max_num
        self.img_size = img_size
        self.no_true_false_equal = no_true_false_equal
        self.no_neg_dir_data = no_neg_dir_data
        self.only_true_data = only_true_data
        self.angle_system = bool(angle_system)
        self.waypoints_dim = waypoints_dim

        # data buffer
        self.dataset = dict()
        for primact_type in primact_types:
            self.dataset[primact_type] = []

        # data features
        self.data_features = data_features  # list

        if self.angle_system:
            self.degree_lower = degree_lower
            self.degree_upper = degree_upper
        else:
            self.degree_lower = 5.73
            self.degree_upper = 25.8

        self.EP_MAX = EP_MAX

        self.curiosityDriven = curiosityDriven
        self.cur_primact_type = cur_primact_type
        if curiosityDriven:
            self.cur_shape_id=cur_shape_id
            self.cur_category=cur_category
            self.cur_cnt_id=cur_cnt_id
            self.cur_primact_type=cur_primact_type
            self.cur_trial_id=cur_trial_id

        self.critic_mode = critic_mode
        self.train_mode = train_mode
        self.affordance_mode = affordance_mode


    def load_data(self, data_list, wp_xyz=False, coordinate_system='world', num_data_uplimit=100000, grasp_replace_upper=1000, num_contactError_uplimit=100000):

        task_uplimit = 20000

        # start loading data
        degree_record = [0 for i in range(8)]
        bar = ProgressBar()
        cnt = 0
        waypoint_none_num = 0
        for i in bar(range(len(data_list))):
            cur_dir = data_list[i]

            positive_cnt = 0

            if self.curiosityDriven:
                cur_category = self.cur_category
                cur_cnt_id = self.cur_cnt_id
                cur_primact_type = self.cur_primact_type
                cur_trial_id = self.cur_trial_id
            else:
                cur_cnt_id = '0'
                cur_primact_type = self.cur_primact_type
                cur_category = None
                cur_trial_id = 1

            for result_idx in range(self.EP_MAX):
                if positive_cnt > num_data_uplimit:  # 为了balance curiosity 不同epoch的data
                    break
                if not os.path.exists(os.path.join(cur_dir, 'result_%d.json' % result_idx)):
                    continue
                with open(os.path.join(cur_dir, 'result_%d.json' % result_idx), 'r') as fin:
                    result_data = json.load(fin)

                    # discard if contact error
                    if result_data['result'] != 'VALID':
                        continue

                    if self.angle_system:
                        task_motion = result_data['actual_task']
                    else:
                        task_motion = result_data['actual_task'] * 180.0 / 3.1415926535

                    if degree_record[int(np.abs(task_motion) // 10) - 1] > task_uplimit:
                        continue            # balance data ！

                    ''' front views '''
                    camera_metadata = result_data['camera_metadata']
                    cam_theta, cam_phi = camera_metadata['theta'], camera_metadata['phi']
                    if cam_theta <= 1/4 * 2 * np.pi or cam_theta >= 3/4 * 2 * np.pi:
                        continue

                    mat44 = np.array(camera_metadata['mat44'], dtype=np.float32)
                    cur_shape_id = result_data['shape_id']


                    # if self.world_coordinate:
                    gripper_direction_world = np.array(result_data['gripper_direction_world'], dtype=np.float32)  # up
                    gripper_forward_direction_world = np.array(result_data['gripper_forward_direction_world'], dtype=np.float32)  # forward
                    position_world = np.array(result_data['position_world'], dtype=np.float32)  # contact point

                    target_part_joint_idx = result_data['target_object_part_joint_id']
                    joint_angles = np.array(result_data['joint_angles'], dtype=np.float32)
                    joint_angle = joint_angles[target_part_joint_idx]

                    task_waypoints = self.get_task_waypoints(result_data)
                    # dense_waypoints = np.array(result_data['dense_waypoints'], dtype=np.float32)
                    if task_waypoints.shape == (1,):     # [None]
                        waypoint_none_num += 1
                        continue
                    dense_waypoints = None

                    ori_pixel_ids = np.array(result_data['pixel_locs'], dtype=np.int32)
                    pixel_ids = np.round(np.array(result_data['pixel_locs'], dtype=np.float32) / 448 * self.img_size).astype(np.int32)

                    success = True

                    # load original data
                    if success:
                        if coordinate_system == 'world':
                            cur_data = (cur_dir, cur_shape_id, cur_category, cur_cnt_id, cur_trial_id, ori_pixel_ids, pixel_ids,
                                        gripper_direction_world, gripper_forward_direction_world, True, True, task_motion, task_motion,
                                        task_waypoints, position_world, result_idx, mat44, None, None, None, joint_angle,
                                        False, dense_waypoints, position_world, coordinate_system, camera_metadata, joint_angles)

                        self.dataset[cur_primact_type].append(cur_data)
                        cnt += 1
                        positive_cnt += 1
                        degree_record[int(np.abs(task_motion) // 10) - 1] += 1

                        # negative
                        if not self.only_true_data:
                            random_idx = 1 - random.randint(0, 1) * 2
                            if coordinate_system == 'world':
                                cur_data = (cur_dir, cur_shape_id, cur_category, cur_cnt_id, cur_trial_id, ori_pixel_ids, pixel_ids,
                                            gripper_direction_world, gripper_forward_direction_world, True, False, task_motion + random_idx * random.uniform(abs(task_motion) * 0.1, self.degree_upper), task_motion,
                                            task_waypoints, position_world, result_idx, mat44, None, None, None, joint_angle,
                                            False, dense_waypoints, position_world, coordinate_system, camera_metadata, joint_angles)

                            self.dataset[cur_primact_type].append(cur_data)
                            cnt += 1
            print('waypoint_none_num: ', waypoint_none_num)
            print('degree_record: ', degree_record, 'sum: ', sum(degree_record))
        print('number of data: ', cnt)

        if self.critic_mode:
            bar = ProgressBar()
            len_positive_data = len(self.dataset[self.cur_primact_type])
            tot_num_contact_error = 0
            for i in bar(range(len(data_list))):
                cur_dir = data_list[i]
                num_contact_error = 0

                if self.curiosityDriven:
                    cur_shape_id = self.cur_shape_id
                    cur_category = self.cur_category
                    cur_cnt_id = self.cur_cnt_id
                    cur_primact_type = self.cur_primact_type
                    cur_trial_id = self.cur_trial_id
                else:
                    cur_cnt_id = '0'
                    cur_primact_type = self.cur_primact_type
                    cur_shape_id = 48855
                    cur_category = None
                    cur_trial_id = 1

                for result_idx in range(self.EP_MAX):
                    if not os.path.exists(os.path.join(cur_dir, 'result_%d.json' % result_idx)):
                        continue
                    with open(os.path.join(cur_dir, 'result_%d.json' % result_idx), 'r') as fin:
                        result_data = json.load(fin)

                        # remain contact error
                        if result_data['result'] != 'CONTACT_ERROR' and result_data['result'] != 'GRASP_ERROR':
                            continue

                        ''' front views '''
                        camera_metadata = result_data['camera_metadata']
                        cam_theta, cam_phi = camera_metadata['theta'], camera_metadata['phi']
                        if cam_theta <= 1/4 * 2 * np.pi or cam_theta >= 3/4 * 2 * np.pi:
                            continue

                        mat44 = np.array(camera_metadata['mat44'], dtype=np.float32)

                        # world_coordinate:
                        gripper_direction_world = np.array(result_data['gripper_direction_world'], dtype=np.float32)  # up
                        gripper_forward_direction_world = np.array(result_data['gripper_forward_direction_world'], dtype=np.float32)  # forward
                        position_world = np.array(result_data['position_world'], dtype=np.float32)  # contact point

                        ori_pixel_ids = np.array(result_data['pixel_locs'], dtype=np.int32)
                        pixel_ids = np.round(np.array(result_data['pixel_locs'], dtype=np.float32) / 448 * self.img_size).astype(np.int32)
                        target_part_joint_idx = result_data['target_object_part_joint_id']
                        joint_angles = np.array(result_data['joint_angles'], dtype=np.float32)
                        joint_angle = joint_angles[target_part_joint_idx]

                        if self.affordance_mode:
                            if self.angle_system:
                                task_motion = result_data['actual_task']
                            else:
                                task_motion = result_data['actual_task'] * 180.0 / 3.1415926535
                            task_waypoints = None
                        else:
                            refer_data = None
                            min_dis = 10000
                            for idx_positive_data in range(len_positive_data):
                                positive_data = self.dataset[cur_primact_type][idx_positive_data]
                                if positive_data[10] == False:  # negative data
                                    continue
                                post_position_world = positive_data[23]
                                post_joint_angle = positive_data[20]
                                dis_ctpt = np.linalg.norm(post_position_world - position_world)
                                dis_joint_angle = np.linalg.norm(post_joint_angle - joint_angle)
                                if dis_ctpt + dis_joint_angle < min_dis:    # find min(dis_ctpt + dis_joint_angle)
                                    min_dis = dis_ctpt + dis_joint_angle
                                    refer_data = positive_data

                            # get waypoints for contact_error data
                            task_motion = refer_data[12]
                            task_waypoints = refer_data[13]
                            # gripper_direction_world = refer_data[7]
                            # gripper_forward_direction_world = refer_data[8]

                        cur_data = (cur_dir, cur_shape_id, cur_category, cur_cnt_id, cur_trial_id, ori_pixel_ids, pixel_ids,
                                    gripper_direction_world, gripper_forward_direction_world, True, False, task_motion, task_motion,
                                    task_waypoints, position_world, result_idx, mat44, None, None, None, joint_angle,
                                    True, None, position_world, coordinate_system, camera_metadata, joint_angles)

                        # eval, load all data with contact/grasp error
                        if self.affordance_mode:
                            self.dataset[cur_primact_type].append(cur_data)
                            num_contact_error += 1
                            tot_num_contact_error += 1
                            if num_contact_error > 1000:
                                break
                        else:
                            if not self.train_mode:
                                self.dataset[cur_primact_type].append(cur_data)
                            # train, find a negative data point and replace it
                            else:
                                if tot_num_contact_error > grasp_replace_upper:
                                    data_idx = random.randint(0, len_positive_data - 1)
                                    while self.dataset[cur_primact_type][data_idx][10] == False:
                                        data_idx = random.randint(0, len_positive_data - 1)
                                    self.dataset[cur_primact_type].append(cur_data)
                                    self.dataset[cur_primact_type].append(self.dataset[cur_primact_type][data_idx])
                                else:
                                    data_idx = random.randint(0, len_positive_data- 1)
                                    while self.dataset[cur_primact_type][data_idx][10] == True:
                                        data_idx = random.randint(0, len_positive_data - 1)
                                    self.dataset[cur_primact_type][data_idx] = cur_data
                            num_contact_error += 1
                            tot_num_contact_error += 1
                            if num_contact_error > num_contactError_uplimit:
                                break
                print('number of contact error: ', tot_num_contact_error)


    def load_data_fromMemory(self, out_info_batch, coordinate_system='world'):
        for i in range(len(out_info_batch)):
            result_data = out_info_batch[i]
            cur_dir = result_data['out_dir']
            cur_shape_id = result_data['shape_id']
            cur_category = result_data['category']
            cur_cnt_id = result_data['cnt_id']
            cur_primact_type = result_data['primact_type']
            cur_trial_id = result_data['trial_id']
            gripper_direction_world = np.array(result_data['gripper_direction_world'], dtype=np.float32)  # up
            gripper_forward_direction_world = np.array(result_data['gripper_forward_direction_world'], dtype=np.float32)  # forward
            position_world = np.array(result_data['position_world'], dtype=np.float32)  # contact point
            if self.angle_system:
                gt_motion = result_data['actual_task']
                task_motion = result_data['task']
            else:
                gt_motion = result_data['actual_task'] * 180.0 / 3.1415926535
                task_motion = result_data['task'] * 180.0 / 3.1415926535
            task_waypoints = self.get_task_waypoints(result_data)
            ori_pixel_ids = np.array(result_data['pixel_locs'], dtype=np.int32)
            pixel_ids = np.round(np.array(result_data['pixel_locs'], dtype=np.float32) / 448 * self.img_size).astype(np.int32)
            result_idx = result_data['epoch']
            camera_metadata = result_data['camera_metadata']
            mat44 = np.array(camera_metadata['mat44'], dtype=np.float32)
            if 'input_pcs' in result_data.keys():
                input_pcs = np.array(result_data['input_pcs'], dtype=np.float32)
                pc_pxids = np.array(result_data['pc_pxids'], dtype=np.float32)
                pc_movables = np.array(result_data['pc_movables'], dtype=np.float32)
            else:
                input_pcs = None
                pc_pxids = None
                pc_movables = None

            if result_data['type'] == 1:
                cur_data = (cur_dir, cur_shape_id, cur_category, cur_cnt_id, cur_trial_id, ori_pixel_ids, pixel_ids,
                            gripper_direction_world, gripper_forward_direction_world, True, False, task_motion, gt_motion,
                            task_waypoints, position_world, result_idx, mat44, input_pcs, pc_pxids, pc_movables, 0,
                            False, None, position_world, coordinate_system, None, None)
                self.dataset[cur_primact_type].append(cur_data)
            if (result_data['type'] == 2) or (result_data['type'] == 3):
                cur_data = (cur_dir, cur_shape_id, cur_category, cur_cnt_id, cur_trial_id, ori_pixel_ids, pixel_ids,
                            gripper_direction_world, gripper_forward_direction_world, True, True, gt_motion, gt_motion,
                            task_waypoints, position_world, result_idx, mat44, input_pcs, pc_pxids, pc_movables, 0,
                            False, None, position_world, coordinate_system, None, None)
                self.dataset[cur_primact_type].append(cur_data)
            # negative
            if result_data['type'] == 3:
                random_idx = 1 - random.randint(0, 1) * 2
                cur_data = (cur_dir, cur_shape_id, cur_category, cur_cnt_id, cur_trial_id, ori_pixel_ids, pixel_ids,
                            gripper_direction_world, gripper_forward_direction_world, True, False, gt_motion + random_idx * random.uniform(abs(gt_motion) * 0.1, self.degree_upper), gt_motion,
                            task_waypoints, position_world, result_idx, mat44, input_pcs, pc_pxids, pc_movables, 0,
                            False, None, position_world, coordinate_system, None, None)
                self.dataset[cur_primact_type].append(cur_data)

    def get_task_waypoints(self, result_data):
        task_waypoints = result_data['waypoints']
        num_step = result_data['num_steps']
        task_waypoints = np.array(task_waypoints)
        if task_waypoints.shape[0] == 0:
            return np.array([None])
        if self.waypoints_dim == 6:
            task_waypoints = task_waypoints[:, :6]
        elif self.waypoints_dim == 3:
            task_waypoints = task_waypoints[:, :3]
        while len(task_waypoints) < num_step - 1:
            task_waypoints = np.concatenate([task_waypoints, [task_waypoints[-1]]])
        return task_waypoints

    def __str__(self):
        strout = '[SAPIENVisionDataset %d] primact_types: %s, img_size: %d\n' % \
                (len(self), ','.join(self.primact_types), self.img_size)
        for primact_type in self.primact_types:
            # strout += '\t<%s> True: %d Task_fail: %d False: %d\n' % (primact_type, len(self.true_data[primact_type]), len(self.task_fail_data[primact_type]), len(self.false_data[primact_type]))
            strout += '\t<%s> Dataset: %d\n' % (primact_type, len(self.dataset[primact_type]))
        return strout

    def __len__(self):
        max_data = 0
        for primact_type in self.primact_types:
            max_data = max(max_data, len(self.dataset[primact_type]))
        return max_data * len(self.primact_types)

    def __getitem__(self, index):
        primact_id = index % len(self.primact_types)
        primact_type = self.primact_types[primact_id]
        index = index // len(self.primact_types)

        cur_dir, shape_id, category, cnt_id, trial_id, ori_pixel_ids, pixel_ids, \
                gripper_direction, gripper_forward_direction, is_original, result, task_motion, gt_motion, \
                task_waypoints, position, result_idx, mat44, input_pcs, pc_pxids, pc_movables, joint_angle, contact_error, \
                dense_waypoints, position_world, coordinate_system, camera_metadata, joint_angles = self.dataset[primact_type][index]

        # grids
        grid_x, grid_y = np.meshgrid(np.arange(448), np.arange(448))
        grid_xy = np.stack([grid_y, grid_x]).astype(np.int32)    # 2 x 448 x 448

        data_feats = ()
        out2 = None
        out3 = None
        for feat in self.data_features:
            if feat == 'img':
                with Image.open(os.path.join(cur_dir, 'rgb.png')) as fimg:
                    out = np.array(fimg.resize((self.img_size, self.img_size)), dtype=np.float32) / 255
                out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                data_feats = data_feats + (out,)
             
            elif feat == 'pcs':
                if type(input_pcs) is np.ndarray:
                    data_feats = data_feats + (torch.from_numpy(input_pcs),)
                else:
                    x, y = ori_pixel_ids[0], ori_pixel_ids[1]
                    with h5py.File(os.path.join(cur_dir, 'cam_XYZA_%d.h5' % result_idx), 'r') as fin:
                        cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                        cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                        cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                    out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
                    with Image.open(os.path.join(cur_dir, 'interaction_mask_%d.png' % result_idx)) as fimg:
                        out3 = (np.array(fimg, dtype=np.float32) > 127)
                    pt = out[x, y, :3]
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
                    idx = idx[:30000-1]
                    pc = pc[idx, :]
                    pcids = pcids[idx, :]
                    out3 = out3[idx]
                    pc = np.vstack([pt, pc])
                    pcids = np.vstack([ptid, pcids])
                    out3 = np.append(True, out3)
                    # normalize to zero-centered
                    pc[:, 0] -= 5
                    if coordinate_system == 'world':
                        pc = (mat44[:3, :3] @ pc.T).T
                    pc = np.array(pc, dtype=np.float32)

                    out = torch.from_numpy(pc).unsqueeze(0)
                    out2 = torch.from_numpy(pcids).unsqueeze(0)
                    out3 = torch.from_numpy(out3).unsqueeze(0).float()
                    data_feats = data_feats + (out,)

            elif feat == 'pc_pxids':
                if type(pc_pxids) is np.ndarray:
                    data_feats = data_feats + (torch.from_numpy(pc_pxids),)
                else:
                    data_feats = data_feats + (out2,)
             
            elif feat == 'pc_movables':
                if type(pc_movables) is np.ndarray:
                    data_feats = data_feats + (torch.from_numpy(pc_movables).float(),)
                else:
                    data_feats = data_feats + (out3,)

            elif feat == 'task_waypoints':
                data_feats = data_feats + (task_waypoints,)
             
            elif feat == 'interaction_mask_small':
                with Image.open(os.path.join(cur_dir, 'interaction_mask_%d.png' % result_idx)) as fimg:
                    out = np.array(fimg.resize((self.img_size, self.img_size)), dtype=np.float32) / 255
                out = (torch.from_numpy(out) > 0.5).float().unsqueeze(0)
                data_feats = data_feats + (out,)
             
            elif feat == 'interaction_mask':
                with Image.open(os.path.join(cur_dir, 'interaction_mask_%d.png' % result_idx)) as fimg:
                    out = np.array(fimg, dtype=np.float32) / 255
                data_feats = data_feats + (out,)

            elif feat == 'gripper_img_target':
                if is_original:
                    png_fn = os.path.join(cur_dir, 'viz_target_pose.png')
                    if not os.path.exists(png_fn):
                        out = torch.ones(1, 3, 448, 448).float()
                    else:
                        with Image.open(os.path.join(cur_dir, 'viz_target_pose.png')) as fimg:
                            out = np.array(fimg, dtype=np.float32) / 255
                            out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                else:
                    out = torch.ones(1, 3, 448, 448).float()
                data_feats = data_feats + (out,)

            elif feat == 'is_original':
                data_feats = data_feats + (is_original,)
            
            elif feat == 'pixel_id':
                out = torch.from_numpy(pixel_ids).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'gripper_direction':
                out = torch.from_numpy(gripper_direction).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'gripper_forward_direction':
                out = torch.from_numpy(gripper_forward_direction).unsqueeze(0)
                data_feats = data_feats + (out,)
             
            elif feat == 'result':
                data_feats = data_feats + (result,)

            elif feat == 'task_motion':
                data_feats = data_feats + (task_motion,)

            elif feat == 'gt_motion':
                data_feats = data_feats + (gt_motion,)

            elif feat == 'position':
                data_feats = data_feats + (position,)

            elif feat == 'contact_error':
                data_feats = data_feats + (contact_error,)

            elif feat == 'epoch_idx':
                data_feats = data_feats + (result_idx,)

            elif feat == 'dense_waypoints':
                out = dense_waypoints
                data_feats = data_feats + (out,)

            elif feat == 'cur_dir':
                data_feats = data_feats + (cur_dir,)

            elif feat == 'shape_id':
                data_feats = data_feats + (shape_id,)

            elif feat == 'primact_type':
                data_feats = data_feats + (primact_type,)
            
            elif feat == 'category':
                data_feats = data_feats + (category,)

            elif feat == 'cnt_id':
                data_feats = data_feats + (cnt_id,)

            elif feat == 'trial_id':
                data_feats = data_feats + (trial_id,)

            elif feat == 'gripper_direction_world':
                out = torch.from_numpy(gripper_direction).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'gripper_forward_direction_world':
                out = torch.from_numpy(gripper_forward_direction).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'position_world':
                data_feats = data_feats + (position_world,)

            elif feat == 'mat44':
                data_feats = data_feats + (mat44,)

            elif feat == 'camera_metadata':
                data_feats = data_feats + (camera_metadata,)

            elif feat == 'joint_angles':
                data_feats = data_feats + (joint_angles,)

            elif feat == 'ori_pixel_ids':
                out = ori_pixel_ids
                data_feats = data_feats + (out,)

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

