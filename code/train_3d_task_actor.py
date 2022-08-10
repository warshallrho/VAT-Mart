"""
    Train the Actionability Prediction Module only
"""

import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
from data_iclr import SAPIENVisionDataset
import utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'blender_utils'))
from tensorboardX import SummaryWriter


def train(conf, train_shape_list, train_data_list, val_data_list):
    # create training and validation datasets and data loaders
    data_features = ['pcs', 'pc_pxids', 'pc_movables', 'gripper_img_target', 'gripper_direction_world', 'gripper_forward_direction_world', \
            'result', 'task_motion', 'gt_motion', 'task_waypoints', 'cur_dir', 'shape_id', 'trial_id', 'is_original', 'position_world']

    ''' input:  task， init position， contact point, waypoint '''
     
    # load network model
    model_def = utils.get_model_module(conf.model_version)

    # create models
    network = model_def.ActorNetwork(conf.feat_dim, num_steps=conf.num_steps, lbd_kl=conf.lbd_kl, lbd_dir=conf.lbd_dir, lbd_recon=conf.lbd_recon, if_noise=bool(conf.actor_noise))
    utils.printout(conf.flog, '\n' + str(network) + '\n')

    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR    TotalLoss   KLLoss  RecLoss   DirLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        # from tensorboardX import SummaryWriter
        train_writer = SummaryWriter(os.path.join(conf.tb_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(conf.tb_dir, 'val'))

    # send parameters to device
    network.to(conf.device)
    utils.optimizer_to_device(network_opt, conf.device)

    # load dataset
    train_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.buffer_max_num, \
            img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal, angle_system=conf.angle_system, EP_MAX=conf.num_train, degree_lower=conf.degree_lower)

    val_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.buffer_max_num, \
            img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal, angle_system=conf.angle_system, EP_MAX=conf.num_eval, degree_lower=conf.degree_lower)

    ### load data for the current epoch
    print("len of train data list", len(train_data_list))
    train_dataset.load_data(train_data_list, wp_xyz=conf.wp_xyz)  # 每个实验路径的list
    utils.printout(conf.flog, str(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True,
                                                   pin_memory=True, \
                                                   num_workers=0, drop_last=True, collate_fn=utils.collate_feats,
                                                   worker_init_fn=utils.worker_init_fn)
    train_num_batch = len(train_dataloader)

    val_dataset.load_data(val_data_list, wp_xyz=conf.wp_xyz)
    utils.printout(conf.flog, str(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False, pin_memory=True, \
            num_workers=0, drop_last=True, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
    val_num_batch = len(val_dataloader)
    print('train_num_batch: %d, val_num_batch: %d' % (train_num_batch, val_num_batch))


    # start training
    start_time = time.time()

    last_train_console_log_step, last_val_console_log_step = None, None

    start_epoch = 0


    # train for every epoch
    for epoch in range(start_epoch, conf.epochs):   # 每个epoch重新获得一次train dataset
        ### collect data for the current epoch
        if epoch > start_epoch:
            utils.printout(conf.flog, f'  [{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} Waiting epoch-{epoch} data ]')

            cur_data_folders = []
            for item in train_data_list:
                item = '/'.join(item.split('/')[:-1])
                if item not in cur_data_folders:
                    cur_data_folders.append(item)
            for cur_data_folder in cur_data_folders:
                with open(os.path.join(cur_data_folder, 'data_tuple_list.txt'), 'w') as fout:
                    for item in train_data_list:
                        if cur_data_folder == '/'.join(item.split('/')[:-1]):
                            fout.write(item.split('/')[-1]+'\n')

        ### print log
        if not conf.no_console_log:
            utils.printout(conf.flog, f'training run {conf.exp_name}')
            utils.printout(conf.flog, header)

        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)

        train_fraction_done = 0.0
        val_fraction_done = 0.0
        val_batch_ind = -1

        ### train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step
            
            # save checkpoint
            if epoch % 5 == 0 and train_batch_ind == 0:
                with torch.no_grad():
                    utils.printout(conf.flog, 'Saving checkpoint ...... ')
                    torch.save(network.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % epoch))
                    torch.save(network_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % epoch))
                    torch.save(network_lr_scheduler.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % epoch))
                    torch.save(train_dataset, os.path.join(conf.exp_dir, 'ckpts', '%d-train_dataset.pth' % epoch))
                    utils.printout(conf.flog, 'DONE')

            # set models to training mode
            network.train()

            # forward pass (including logging)
            losses = actor_forward(batch=batch, data_features=data_features, network=network, conf=conf, is_val=False, \
                    step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time, \
                    log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer, lr=network_opt.param_groups[0]['lr'])
            total_loss = losses['tot']
            # optimize one step
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()
            network_lr_scheduler.step()

            # validate one batch
            val_cnt = 0
            # total_loss, total_precision, total_recall, total_Fscore, total_accu = 0, 0, 0, 0, 0
            while val_fraction_done <= train_fraction_done and val_batch_ind+1 < val_num_batch:
                val_cnt += 1
                val_batch_ind, val_batch = next(val_batches)

                val_fraction_done = (val_batch_ind + 1) / val_num_batch
                val_step = (epoch + val_fraction_done) * train_num_batch - 1

                log_console = not conf.no_console_log and (last_val_console_log_step is None or \
                        val_step - last_val_console_log_step >= conf.console_log_interval)
                if log_console:
                    last_val_console_log_step = val_step

                # set models to evaluation mode
                network.eval()

                with torch.no_grad():
                    # forward pass (including logging)
                    total_loss = actor_forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
                            step=val_step, epoch=epoch, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
                            log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=val_writer, lr=network_opt.param_groups[0]['lr'])
                    # total_loss += loss
                    # total_precision += precision
                    # total_recall += recall
                    # total_Fscore += Fscore
                    # total_accu += accu

            # if val_cnt > 0:
            #     avg_loss = total_loss / val_cnt
            #     avg_precision = total_precision / val_cnt
            #     avg_recall = total_recall / val_cnt
            #     avg_Fscore = total_Fscore / val_cnt
            #     avg_accu = total_accu / val_cnt
            #     print("total_loss: %f, precision: %f, recall: %f, Fscore: %f, accuracy: %f" % (avg_loss, avg_precision, avg_recall, avg_Fscore, avg_accu))


def actor_forward(batch, data_features, network, conf, \
            is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
            log_console=False, log_tb=False, tb_writer=None, lr=None):
    # prepare input
    input_pcs = torch.cat(batch[data_features.index('pcs')], dim=0).to(conf.device)  # B x 3N x 3   # point cloud
    batch_size = input_pcs.shape[0]

    input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(-1)  # BN
    input_pcid2 = furthest_point_sample(input_pcs, conf.num_point_per_shape).long().reshape(-1)  # BN
    input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)

    input_dirs1 = torch.cat(batch[data_features.index('gripper_direction_world')], dim=0).to(conf.device)  # B x 3 # up作为feature
    input_dirs2 = torch.cat(batch[data_features.index('gripper_forward_direction_world')], dim=0).to(conf.device)  # B x 3   # forward
    actual_motion = torch.Tensor(batch[data_features.index('gt_motion')]).to(conf.device)  # B  # 度数
    task_waypoints = torch.Tensor(batch[data_features.index('task_waypoints')]).to(conf.device)     # 取 waypoint, 4*3 (初始一定是(0,0,0), gripper坐标系)
    task_traj = torch.cat([torch.cat([input_dirs1, input_dirs2], dim=0).view(conf.batch_size, 2, 3), task_waypoints], dim=1).view(conf.batch_size, conf.num_steps + 1, 3)  # up和forward两个方向拼起来 + waypoints
    contact_point = torch.Tensor(batch[data_features.index('position_world')]).to(conf.device)

    losses = network.get_loss(input_pcs, actual_motion, task_traj, contact_point)  # B x 2, B x F x N
    kl_loss = losses['kl']
    total_loss = losses['tot']
    recon_loss = losses['recon']
    dir_loss = losses['dir']

    # display information
    data_split = 'train'
    if is_val:
        data_split = 'val'

    with torch.no_grad():
        # log to console
        if log_console:
            utils.printout(conf.flog, \
                           f'''{strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} '''
                           f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                           f'''{data_split:^10s} '''
                           f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                           f'''{100. * (1 + batch_ind + num_batch * epoch) / (num_batch * conf.epochs):>9.1f}%      '''
                           f'''{lr:>5.2E} '''
                           f'''{total_loss.item():>10.5f}'''
                           f'''{kl_loss.item():>10.5f}'''
                           f'''{recon_loss.item():>10.5f}'''
                           f'''{dir_loss.item():>10.5f}'''
                           )
            conf.flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('total_loss', total_loss.item(), step)
            tb_writer.add_scalar('kl_loss', kl_loss.item(), step)
            tb_writer.add_scalar('dir_loss', dir_loss.item(), step)
            tb_writer.add_scalar('recon_loss', recon_loss.item(), step)
            tb_writer.add_scalar('lr', lr, step)

        return losses


if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()
    
    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--model_version', type=str, help='model def file')
    parser.add_argument('--primact_type', type=str, help='the primact type')
    parser.add_argument('--category_types', type=str, help='list all categories [Default: None, meaning all 10 categories]', default=None)
    parser.add_argument('--data_dir_prefix', type=str, help='data directory')
    parser.add_argument('--offline_data_dir', type=str, help='data directory')
    parser.add_argument('--val_data_dir', type=str, help='data directory')
    parser.add_argument('--train_shape_fn', type=str, help='training shape file that indexs all shape-ids')
    parser.add_argument('--ins_cnt_fn', type=str, help='a file listing all category instance count')

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='../logs', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--resume', action='store_true', default=False, help='resume if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--num_point_per_shape', type=int, default=10000)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--no_true_false_equal', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')

    # training parameters
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--buffer_max_num', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=500)
    parser.add_argument('--sample_succ', action='store_true', default=False)
    parser.add_argument('--angle_system', type=int, default=0)
    parser.add_argument('--num_train', type=int, default=2000)
    parser.add_argument('--num_eval', type=int, default=200)
    parser.add_argument('--degree_lower', type=int, default=15)
    parser.add_argument('--actor_noise', type=int, default=1)
    parser.add_argument('--wp_xyz', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')

    # loss weights
    parser.add_argument('--lbd_kl', type=float, default=1.0)
    parser.add_argument('--lbd_dir', type=float, default=1.0)
    parser.add_argument('--lbd_recon', type=float, default=1.0)

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10, help='number of optimization steps beween console log prints')

    # visu
    parser.add_argument('--num_batch_every_visu', type=int, default=1, help='num batch every visu')
    parser.add_argument('--num_epoch_every_visu', type=int, default=10, help='num epoch every visu')
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')

    # parse args
    conf = parser.parse_args()


    ### prepare before training
    # make exp_name
    conf.exp_name = f'exp-{conf.model_version}-{conf.primact_type}-{conf.category_types}-{conf.exp_suffix}'

    if conf.overwrite and conf.resume:
        raise ValueError('ERROR: cannot specify both --overwrite and --resume!')

    # mkdir exp_dir; ask for overwrite if necessary; or resume
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    print('exp_dir: ', conf.exp_dir)
    conf.tb_dir = os.path.join(conf.exp_dir, 'tb')
    if os.path.exists(conf.exp_dir):
        if not conf.resume:
            if not conf.overwrite:
                response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
                if response != 'y':
                    exit(1)
            shutil.rmtree(conf.exp_dir)
    else:
        if conf.resume:
            raise ValueError('ERROR: no training run named %s to resume!' % conf.exp_name)
    if not conf.resume:
        os.mkdir(conf.exp_dir)
        os.mkdir(conf.tb_dir)
        os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))
        if not conf.no_visu:
            os.mkdir(os.path.join(conf.exp_dir, 'val_visu'))

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    if not conf.resume:
        torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))

    # file log
    if conf.resume:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'a+')
    else:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    conf.flog = flog

    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')

    # set training device
    device = torch.device(conf.device)
    utils.printout(flog, f'Using device: {conf.device}\n')
    conf.device = device
    
    # parse params
    utils.printout(flog, 'primact_type: %s' % str(conf.primact_type))

    if conf.category_types is None:
        conf.category_types = ['Box', 'Door', 'Faucet', 'Kettle', 'Microwave', 'Refrigerator', 'StorageFurniture', 'Switch', 'TrashCan', 'Window']
    else:
        conf.category_types = conf.category_types.split(',')
    utils.printout(flog, 'category_types: %s' % str(conf.category_types))
    
    # read cat2freq
    conf.cat2freq = dict()
    with open(conf.ins_cnt_fn, 'r') as fin:
        for l in fin.readlines():
            category, _, freq = l.rstrip().split()
            conf.cat2freq[category] = int(freq)
    utils.printout(flog, str(conf.cat2freq))

    # read train_shape_fn
    train_shape_list = []


    train_data_list = []
    for root, dirs, files in os.walk(conf.offline_data_dir):
        for dir in dirs:
            train_data_list.append(os.path.join(conf.offline_data_dir, dir))
    utils.printout(flog, 'len(train_data_list): %d' % len(train_data_list))

    val_data_list = []
    for root, dirs, files in os.walk(conf.val_data_dir):
        for dir in dirs:
            val_data_list.append(os.path.join(conf.val_data_dir, dir))
    utils.printout(flog, 'len(train_data_list): %d' % len(val_data_list))

     
    ### start training
    print('train_data_list: ', train_data_list[0])
    train(conf, train_shape_list, train_data_list, val_data_list)


    ### before quit
    # close file log
    flog.close()

