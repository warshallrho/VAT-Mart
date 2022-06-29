import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


class Critic(nn.Module):
    def __init__(self, feat_dim, task_feat_dim=32, traj_feat_dim=256, cp_feat_dim=32, num_steps=10):
        super(Critic, self).__init__()

        self.mlp_task = nn.Linear(1, task_feat_dim)

        self.mlp_traj = nn.Linear(num_steps * 6, traj_feat_dim)
        self.mlp_cp = nn.Linear(3, cp_feat_dim) # contact point

        self.mlp1 = nn.Linear(feat_dim + traj_feat_dim + task_feat_dim + cp_feat_dim, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, 1)

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')
        self.num_steps = num_steps

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, pixel_feats, task, traj, contact_point):
        batch_size = traj.shape[0]
        task = task.view(-1, 1)
        traj = traj.view(batch_size, self.num_steps * 6)
        task_feat = self.mlp_task(task)
        traj_feats = self.mlp_traj(traj)
        cp_feats = self.mlp_cp(contact_point)
        net = torch.cat([pixel_feats, task_feat, traj_feats, cp_feats], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net).squeeze(-1)
        return net

    def forward_n(self, pixel_feats, task, traj, contact_point, rvs):
        batch_size = pixel_feats.shape[0]
        task = task.view(-1, 1)
        traj = traj.view(batch_size * rvs, self.num_steps * 6)
        task_feat = self.mlp_task(task)
        traj_feats = self.mlp_traj(traj)
        cp_feats = self.mlp_cp(contact_point)
        pixel_feats = pixel_feats.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)
        task_feat = task_feat.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)
        cp_feats = cp_feats.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)
        net = torch.cat([pixel_feats, task_feat, traj_feats, cp_feats], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net).squeeze(-1)
        return net

    # cross entropy loss
    def get_ce_loss(self, pred_logits, gt_labels):
        loss = self.BCELoss(pred_logits, gt_labels.float())
        return loss

    # cross entropy loss
    def get_l1_loss(self, pred_logits, gt_labels):
        loss = self.L1Loss(pred_logits, gt_labels)
        return loss


class Network(nn.Module):
    def __init__(self, feat_dim, num_steps):
        super(Network, self).__init__()

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})

        self.critic = Critic(feat_dim, num_steps=num_steps)

    # pcs: B x N x 3 (float), with the 0th point to be the query point
    # pred_result_logits: B, whole_feats: B x F x N
    def forward(self, pcs, task, traj, contact_point):
        pcs[:, 0] = contact_point
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]

        # input_queries = torch.cat([dirs1, dirs2], dim=1)

        pred_result_logits = self.critic(net, task, traj, contact_point)

        return pred_result_logits, whole_feats

    def forward_n(self, pcs, task, traj, contact_point, rvs):
        # pcs[:, 0] = contact_point
        batch_size = task.shape[0]
        pcs[:, 0] = contact_point
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]

        # input_queries = torch.cat([dirs1, dirs2], dim=1)

        pred_result_logits = self.critic.forward_n(net, task, traj, contact_point, rvs=rvs)

        return pred_result_logits, whole_feats

    def inference_critic_score(self, pcs, task, traj):
        # pcs[:, 0] = contact_point
        batch_size = pcs.shape[0]
        pt_size = pcs.shape[1]
        contact_point = pcs.view(batch_size * pt_size, -1)
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        net = whole_feats.permute(0, 2, 1).reshape(batch_size * pt_size, -1)

        task = task.unsqueeze(1).repeat(1, pt_size, 1).view(batch_size * pt_size, 1)

        traj = traj.repeat(batch_size * pt_size, 1, 1)

        pred_result_logits = self.critic.forward(net, task, traj, contact_point)
        pred_result_logits = torch.sigmoid(pred_result_logits)

        return pred_result_logits
