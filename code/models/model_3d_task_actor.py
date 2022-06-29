import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


def KL(mu, logvar):
    mu = mu.view(mu.shape[0], -1)
    logvar = logvar.view(logvar.shape[0], -1)
    loss = 0.5 * torch.sum(mu * mu + torch.exp(logvar) - 1 - logvar, 1)
    # high star implementation
    # torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
    loss = torch.mean(loss)
    return loss


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
    def __init__(self, feat_dim, task_feat_dim=32, traj_feat_dim=256, cp_feat_dim=32, num_steps=10):
        super(ActorEncoder, self).__init__()

        self.mlp1 = nn.Linear(feat_dim + traj_feat_dim + task_feat_dim + cp_feat_dim, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, feat_dim)
        self.get_mu = nn.Linear(feat_dim, feat_dim)
        self.get_logvar = nn.Linear(feat_dim, feat_dim)

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')

        self.z_dim = feat_dim

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, traj_feats, pixel_feats, f_task, f_ctpt):
        net = torch.cat([traj_feats, pixel_feats, f_task, f_ctpt], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net)
        mu = self.get_mu(net)
        logvar = self.get_logvar(net)
        noise = torch.Tensor(torch.randn(*mu.shape)).cuda()
        z = mu + torch.exp(logvar / 2) * noise
        return z, mu, logvar


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


class ActorDecoder(nn.Module):
    def __init__(self, feat_dim, task_feat_dim=32, traj_feat_dim=128, cp_feat_dim=32, num_steps=10):
        super(ActorDecoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + task_feat_dim + traj_feat_dim + cp_feat_dim, 512),
            nn.Linear(512, 256),
            nn.Linear(256, num_steps * 6)
        )
        self.num_steps = num_steps

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, z_all, f_p, f_task, f_ctpt):
        batch_size = z_all.shape[0]
        x = torch.cat([z_all, f_p, f_task, f_ctpt], dim=-1)
        x = self.mlp(x)
        x = x.view(batch_size, self.num_steps, 6)
        return x


class ActorNetwork(nn.Module):
    def __init__(self, feat_dim, task_feat_dim=32, traj_feat_dim=128, cp_feat_dim=32, dir_feat_dim=32, num_steps=10, lbd_kl=1.0, lbd_recon=1.0, lbd_dir=1.0):
        super(ActorNetwork, self).__init__()

        self.z_dim = feat_dim

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})


        self.dir_encoder = nn.Linear(6, dir_feat_dim)
        self.traj_encoder = TrajEncoder(traj_feat_dim=128, num_steps=num_steps)
        self.all_encoder = ActorEncoder(feat_dim, traj_feat_dim=128, num_steps=num_steps)
        self.decoder = ActorDecoder(feat_dim, traj_feat_dim=128, num_steps=num_steps)
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')

        self.mlp_task = nn.Linear(1, task_feat_dim)

        self.mlp_traj = nn.Linear(num_steps * 6, traj_feat_dim)
        self.mlp_cp = nn.Linear(3, cp_feat_dim) # contact point

        self.num_steps = num_steps

        self.lbd_kl = lbd_kl
        self.lbd_recon = lbd_recon
        self.lbd_dir = lbd_dir

    # input sz bszx3x2
    def bgs(self, d6s):
        bsz = d6s.shape[0]
        b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
        a2 = d6s[:, :, 1]
        b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

    # batch geodesic loss for rotation matrices
    def bgdR(self, Rgts, Rps):
        Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
        Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1) #batch trace
        # necessary or it might lead to nans and the likes
        theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
        return torch.acos(theta)

    # 6D-Rot loss
    # input sz bszx6
    def get_6d_rot_loss(self, pred_6d, gt_6d):
        pred_Rs = self.bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        gt_Rs = self.bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        theta = self.bgdR(gt_Rs, pred_Rs)
        return theta

    # pcs: B x N x 3 (float), with the 0th point to be the query point
    # pred_result_logits: B, whole_feats: B x F x N
    def forward(self, pcs, task, traj, contact_point):
        # pcs[:, 0] = contact_point
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]
        f_ctpt = self.mlp_cp(contact_point)
        task = task.view(-1, 1)
        f_task = self.mlp_task(task)

        z_traj = self.traj_encoder(traj)
        z_all, mu, logvar = self.all_encoder(z_traj, net, f_task, f_ctpt)
        recon_traj = self.decoder(z_all, net, f_task, f_ctpt)

        return recon_traj, mu, logvar

    def sample(self, pcs, task, contact_point):
        batch_size = task.shape[0]
        f_task = self.mlp_task(task)
        f_ctpt = self.mlp_cp(contact_point)
        z_all = torch.Tensor(torch.randn(batch_size, self.z_dim)).cuda()

        pcs[:, 0] = contact_point
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]

        recon_traj = self.decoder(z_all, net, f_task, f_ctpt)
        recon_dir = recon_traj[:, 0, :]
        recon_dir = recon_dir.reshape(-1, 3, 2)
        recon_dir = self.bgs(recon_dir)
        recon_wps = recon_traj[:, 1:, :]

        return recon_dir, recon_wps

    def sample_n(self, pcs, task, contact_point, rvs=100):
        batch_size = task.shape[0]
        # ipdb.set_trace()
        task = task.view(-1, 1)
        f_task = self.mlp_task(task)
        f_ctpt = self.mlp_cp(contact_point)
        z_all = torch.Tensor(torch.randn(batch_size * rvs, self.z_dim)).cuda()

        pcs[:, 0] = contact_point
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]
        net = net.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)
        f_task = f_task.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)
        f_ctpt = f_ctpt.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)

        recon_traj = self.decoder(z_all, net, f_task, f_ctpt)

        return recon_traj

    def get_loss(self, pcs, task, traj, contact_point):
        batch_size = traj.shape[0]
        recon_traj, mu, logvar = self.forward(pcs, task, traj, contact_point)
        recon_dir = recon_traj[:, 0, :]
        recon_wps = recon_traj[:, 1:, :]
        input_dir = traj[:, 0, :]
        input_wps = traj[:, 1:, :]
        recon_loss = self.L1Loss(recon_wps.view(batch_size, (self.num_steps - 1) * 6), input_wps.view(batch_size, (self.num_steps - 1) * 6))
        recon_loss = recon_loss.mean()

        dir_loss = self.get_6d_rot_loss(recon_dir, input_dir)
        dir_loss = dir_loss.mean()
        kl_loss = KL(mu, logvar)
        losses = {}
        losses['kl'] = kl_loss
        losses['recon'] = recon_loss
        losses['dir'] = dir_loss
        losses['tot'] = kl_loss * self.lbd_kl + recon_loss * self.lbd_recon + dir_loss * self.lbd_dir

        return losses

    def get_loss_test_rotation(self, pcs, task, traj, contact_point):
        batch_size = traj.shape[0]
        recon_traj, mu, logvar = self.forward(pcs, task, traj, contact_point)
        recon_dir = recon_traj[:, 0, :]
        recon_wps = recon_traj[:, 1:, :]
        input_dir = traj[:, 0, :]
        input_wps = traj[:, 1:, :]
        recon_xyz_loss = self.L1Loss(recon_wps[:, :, 0:3].contiguous().view(batch_size, (self.num_steps - 1) * 3), input_wps[:, :, 0:3].contiguous().view(batch_size, (self.num_steps - 1) * 3))
        recon_rotation_loss = self.L1Loss(recon_wps[:, :, 3:6].contiguous().view(batch_size, (self.num_steps - 1) * 3), input_wps[:, :, 3:6].contiguous().view(batch_size, (self.num_steps - 1) * 3))
        recon_loss = recon_xyz_loss.mean() + recon_rotation_loss.mean() * 100

        dir_loss = self.get_6d_rot_loss(recon_dir, input_dir)
        dir_loss = dir_loss.mean()
        kl_loss = KL(mu, logvar)
        losses = {}
        losses['kl'] = kl_loss
        losses['recon'] = recon_loss
        losses['dir'] = dir_loss
        losses['recon_xyz'] = recon_xyz_loss.mean()
        losses['recon_rotation'] = recon_rotation_loss.mean()
        losses['tot'] = kl_loss * self.lbd_kl + recon_loss * self.lbd_recon + dir_loss * self.lbd_dir

        return losses

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

