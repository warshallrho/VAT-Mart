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


class ActorEncoder(nn.Module):
    def __init__(self, feat_dim, task_feat_dim=32, cp_feat_dim=32):
        super(ActorEncoder, self).__init__()

        self.mlp1 = nn.Linear(feat_dim + task_feat_dim + cp_feat_dim, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, feat_dim)
        self.mlp3 = nn.Linear(feat_dim, 1)

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, pixel_feats, f_task, f_ctpt):
        net = torch.cat([pixel_feats, f_task, f_ctpt], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = F.relu(self.mlp2(net))
        net = self.mlp3(net)
        return net


class TrajEncoder(nn.Module):
    def __init__(self, traj_feat_dim, num_steps=10):
        super(TrajEncoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_steps * 6, 128),
            nn.Linear(128, 128),
            nn.Linear(128, traj_feat_dim)
        )

        self.num_steps = num_steps

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp(x.view(batch_size, self.num_steps * 6))
        return x

class ActionScore(nn.Module):
    def __init__(self, feat_dim, task_feat_dim=32, cp_feat_dim=32, topk=5):
        super(ActionScore, self).__init__()

        self.z_dim = feat_dim
        self.topk = topk

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})

        self.all_encoder = ActorEncoder(feat_dim)
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')
        self.mlp_task = nn.Linear(1, task_feat_dim)
        self.mlp_cp = nn.Linear(3, cp_feat_dim) # contact point

    # pcs: B x N x 3 (float), with the 0th point to be the query point
    # pred_result_logits: B, whole_feats: B x F x N
    def forward(self, pcs, task, contact_point):
        # pcs[:, 0] = contact_point
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]
        f_ctpt = self.mlp_cp(contact_point)
        task = task.view(-1, 1)
        f_task = self.mlp_task(task)
        score = torch.sigmoid(self.all_encoder(net, f_task, f_ctpt))

        return score

    def inference_action_score(self, pcs, task):
        # pcs[:, 0] = contact_point
        batch_size = pcs.shape[0]
        pt_size = pcs.shape[1]
        contact_point = pcs.view(batch_size * pt_size, -1)
        f_ctpt = self.mlp_cp(contact_point)
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        net = whole_feats.permute(0, 2, 1).reshape(batch_size*pt_size, -1)

        task = task.view(-1, 1)
        f_task = self.mlp_task(task)
        f_task = f_task.repeat(pt_size, 1)
        score = torch.sigmoid(self.all_encoder(net, f_task, f_ctpt))

        return score

    def get_loss(self, pcs, task, contact_point, actor, critic, rvs=100):
        batch_size = pcs.shape[0]
        with torch.no_grad():
            traj = actor.sample_n(pcs, task, contact_point, rvs=rvs)
        with torch.no_grad():
            gt_score = torch.sigmoid(critic.forward_n(pcs, task, traj, contact_point, rvs=rvs)[0])
            gt_score = gt_score.view(batch_size, rvs, 1).topk(k=self.topk, dim=1)[0].mean(dim=1).view(-1)
        score = self.forward(pcs, task, contact_point)
        loss = self.L1Loss(score.view(-1), gt_score).mean()

        return loss

    def inference_whole_pc(self, feats, dirs1, dirs2):
        num_pts = feats.shape[-1]
        batch_size = feats.shape[0]

        feats = feats.permute(0, 2, 1)  # B x N x F
        feats = feats.reshape(batch_size*num_pts, -1)

        input_queries = torch.cat([dirs1, dirs2], dim=-1)
        input_queries = input_queries.unsqueeze(dim=1).repeat(1, num_pts, 1)
        input_queries = input_queries.reshape(batch_size*num_pts, -1)

        pred_result_logits = self.critic(feats, input_queries)

        soft_pred_results = torch.sigmoid(pred_result_logits)
        soft_pred_results = soft_pred_results.reshape(batch_size, num_pts)

        return soft_pred_results

    def inference(self, pcs, dirs1, dirs2):
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]

        input_queries = torch.cat([dirs1, dirs2], dim=1)

        pred_result_logits = self.critic(net, input_queries)

        pred_results = (pred_result_logits > 0)

        return pred_results

