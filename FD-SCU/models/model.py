import sys

from scipy.spatial import KDTree

sys.path.append("..")
from torchsparse import SparseTensor
from torchsparse import nn as spnn
import torchsparse.nn.functional as F
from utils import get_indices, upsampling_chunk, DCT1d_matrix
import torch

import numpy as np
from torch import nn

RGB_to_YUV = torch.tensor([[0.299, 0.587, 0.114],
                           [-0.14713, -0.28886, 0.436],
                           [0.615, -0.51499, -0.10001]]).float().cuda()


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super(SEBlock, self).__init__()

        self.fc1 = nn.Linear(input_dim * 2, input_dim // reduction)
        self.fc2 = nn.Linear(input_dim // reduction, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=0, keepdim=True)
        max_pool, _ = torch.max(x, dim=0, keepdim=True)

        squeeze = torch.cat([avg_pool, max_pool], dim=1)
        excitation = torch.nn.functional.relu(self.fc1(squeeze))
        excitation = self.fc2(excitation)

        excitation = self.sigmoid(excitation)

        out = x * excitation
        return out

class SparseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_res=True, use_bias=True,dilation2=1,dilation3=1):
        super().__init__()
        self.use_res = use_res
        self.main1 = nn.Sequential(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        bias=use_bias,
                        dilation=1),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        bias=use_bias,
                        dilation=1),
            spnn.BatchNorm(out_channels)
        )

        self.main2 = nn.Sequential(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        bias=use_bias,
                        dilation=dilation2),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        bias=use_bias,
                        dilation=dilation2),
            spnn.BatchNorm(out_channels)
        )

        self.main3 = nn.Sequential(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        bias=use_bias,
                        dilation=dilation3),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        bias=use_bias,
                        dilation=dilation3),
            spnn.BatchNorm(out_channels)
        )
        self.relu = spnn.ReLU(True)

    def forward(self, x: SparseTensor) -> SparseTensor:

        x1 = self.main1(x)
        x2 = self.main2(x)
        x3 = self.main3(x)
        y = SparseTensor(coords=x.C,feats=torch.cat([x1.F,x2.F,x3.F],dim=1))
        if self.use_res:
            y = self.relu(y + x)
        else:
            y = self.relu(y)
        return y


class FDSCU(nn.Module):
    def __init__(self, voxel_size, bolck_size, num_channel, num_block, global_res=True, use_bias=True):

        super(FDSCU, self).__init__()

        self.voxel_size = voxel_size
        self.global_res = global_res
        self.block_size = bolck_size

        X1 = DCT1d_matrix(self.block_size)
        X = torch.kron(X1, torch.kron(X1, X1))
        self.register_buffer('X', X)


        self.sigma = 0.5

        self.color_block = nn.Sequential()
        self.color_block.append(SparseBlock(in_channels=73, out_channels=20, kernel_size=1,use_res=False, use_bias=use_bias,dilation2=1,dilation3=1))
        for i in range(num_block - 1):
            self.color_block.append(SparseBlock(in_channels=60, out_channels=20, use_res=True, use_bias=use_bias, dilation2=2,dilation3=3))

        self.out_block_3d = nn.Sequential(
            spnn.Conv3d(in_channels=num_channel * 2 - 1, out_channels=num_channel * 4, kernel_size=3, stride=1, bias=True),
            spnn.BatchNorm(num_channel * 4),
            spnn.ReLU(inplace=True),
            spnn.Conv3d(in_channels=num_channel * 4, out_channels=num_channel * 2, kernel_size=2, stride=1, bias=True),
            spnn.BatchNorm(num_channel * 2),
            spnn.ReLU(inplace=True),
            spnn.Conv3d(in_channels=num_channel * 2, out_channels=3, kernel_size=1, stride=1, bias=True),
            spnn.BatchNorm(3)
        )

        self.edge_block = nn.Sequential(
            nn.Linear(in_features=959 + 48,out_features=num_channel * 32,bias=True),
            nn.ReLU(),
            nn.Linear(in_features=num_channel * 32, out_features=num_channel * 16, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=num_channel * 16, out_features=num_channel * 8, bias=True),
            nn.ReLU()
        )

        self.edge_block2 = nn.Sequential(
            nn.Linear(in_features=num_channel, out_features=num_channel // 2, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=num_channel // 2, out_features=3, bias=True),
        )

        self.make = nn.Sequential(
            nn.Linear(in_features=num_channel * 2 - 3, out_features=num_channel * 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=num_channel * 4, out_features=num_channel * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=num_channel * 2, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, sparse_lr: SparseTensor, points_hr: torch.Tensor, batch_size, idx_lr2hr=None):
        block_size = self.block_size
        X = self.X
        grid_res = self.block_size
        range_arr = torch.arange(grid_res, dtype=torch.float32, device='cuda')
        gz, gy, gx = torch.meshgrid(range_arr, range_arr, range_arr, indexing='ij')
        base_grid = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3) * (block_size / grid_res)
        sparse_lr_feats = sparse_lr.feats
        if idx_lr2hr is None:
            quant_hr = torch.cat([points_hr[:, :3] / self.voxel_size, points_hr[:, -1].view(-1, 1)], 1)
            quant_hr = torch.floor(quant_hr).int()
            hash_hr = F.sphash(quant_hr)
            _, idx_lr2hr = torch.unique(hash_hr, sorted=False, return_inverse=True)  # (N_hr)
        sparse_hr_feats = sparse_lr_feats[idx_lr2hr]
        lrcoord = sparse_lr.C
        # RGB -> YUV

        lrcolor = sparse_lr.feats * 255
        lrcolor = lrcolor @ RGB_to_YUV.T

        HF_ALL = []
        LF_ALL = []
        edge_list = []

        indice1 = get_indices(4)  # shape (N_low_Y,)
        indice2 = get_indices(3)  # shape (N_low_UV,)

        full_idx = torch.arange(343, device='cuda')

        mask1 = torch.ones(343, dtype=torch.bool, device='cuda')
        mask1[indice1] = False
        remain1 = full_idx[mask1]

        mask2 = torch.ones(343, dtype=torch.bool, device='cuda')
        mask2[indice2] = False
        remain2 = full_idx[mask2]

        for i in range(batch_size):
            batch_mask = (lrcoord[:, 3] == i)
            points = lrcoord[batch_mask][:, :3].float()  # (N_points, 3)

            find = torch.where(batch_mask)[0]
            colors_batch = lrcolor[find]  # (N_points, 3)


            block_coords = torch.floor(points / block_size).int()


            unique_blocks, point_to_block_idx = torch.unique(block_coords, sorted=False, return_inverse=True, dim=0)
            num_blocks = unique_blocks.shape[0]


            block_origins = unique_blocks.float() * block_size
            all_grid_points = block_origins.unsqueeze(1) + base_grid.unsqueeze(0)


            points_np = points.cpu().numpy()
            kdtree = KDTree(points_np)


            flat_grid_points = all_grid_points.reshape(-1, 3).cpu().numpy()

            _, nn_indices_flat = kdtree.query(flat_grid_points, k=1)

            nn_indices = torch.from_numpy(nn_indices_flat).long().to(points.device)


            gathered_colors = colors_batch[nn_indices]


            dct_input = gathered_colors.reshape(num_blocks, 343, 3).permute(0, 2, 1)

            feat_dct = torch.matmul(dct_input, X.T)
            feat_dct = torch.abs(feat_dct)


            row_sum = feat_dct.sum(dim=2, keepdim=True)
            feat_dct = feat_dct / (row_sum + 1e-8)

            Y = feat_dct[:, 0, :]
            U = feat_dct[:, 1, :]
            V = feat_dct[:, 2, :]

            Y_lf = Y[:, indice1]
            U_lf = U[:, indice2]
            V_lf = V[:, indice2]

            # HF Components
            Y_hf = Y[:, remain1]
            U_hf = U[:, remain2]
            V_hf = V[:, remain2]


            LF_blocks = torch.cat([Y_lf[:, 0:50], U_lf[:, 0:10], V_lf[:, 0:10]], dim=1)  # (Num_Blocks, 70)

            HF_blocks = torch.cat([
                Y_lf[:, 50:], Y_hf,
                U_lf[:, 10:], U_hf,
                V_lf[:, 10:], V_hf
            ], dim=1)

            HF_batch = HF_blocks[point_to_block_idx]
            LF_batch = LF_blocks[point_to_block_idx]

            HF_ALL.append(HF_batch)
            LF_ALL.append(LF_batch)

            k = 16

            color_hr = sparse_hr_feats[find, :]  # (N_points, C)
            color_repeated = color_hr.unsqueeze(1).repeat(1, k, 1)  # (N, K, C)

            _, idxx = kdtree.query(points_np, k + 1)
            idxx = torch.from_numpy(idxx[:, 1:].astype(np.int64)).to(device='cuda')  # (N, K)


            color_select = color_hr[idxx, :]  # (N, K, C)

            color_diff = color_repeated - color_select
            N_pts, K_knn, C_feat = color_diff.shape
            color_edge = color_diff.reshape(N_pts, K_knn * C_feat)
            edge_list.append(color_edge)

        edge = torch.cat(edge_list, dim=0)
        HF_ALL = torch.cat(HF_ALL, dim=0)
        HF_ALL = torch.cat([HF_ALL, edge], dim=1)
        LF_ALL = torch.cat(LF_ALL, dim=0)


        refine = self.edge_block(HF_ALL)
        feat_refine = upsampling_chunk(refine, idx_lr2hr, K_fixed=8)
        feat_refine = self.edge_block2(feat_refine)

        sparse_lr_out = SparseTensor(coords=sparse_lr.C, feats=torch.cat([sparse_lr.feats, LF_ALL], dim=1))

        color_feat_low = self.color_block(sparse_lr_out).feats
        color_feat = color_feat_low[idx_lr2hr]

        se_block = SEBlock(input_dim=60).cuda()
        color_feat = se_block(color_feat)


        sigma = color_feat_low.std(dim=0, unbiased=False).unsqueeze(0)
        sigma = sigma * sigma
        col_sigma = torch.full((1, 1), self.sigma).cuda()
        get_mu_sigma = torch.cat((col_sigma, sigma), dim=1)
        sigma_new = self.make(get_mu_sigma)
        self.sigma = sigma_new.item()
        noise = torch.randn_like(sparse_hr_feats) * sigma_new.item()
        noized = torch.relu(sparse_hr_feats + (noise * 0.1))
        color_feat = torch.cat([color_feat, noized], dim=1)

        out = self.out_block_3d(SparseTensor(coords=points_hr, feats=color_feat)).feats
        out = out + feat_refine
        if self.global_res:
            out = out + sparse_lr_feats[idx_lr2hr, :]
        if not self.training:
            out = torch.clamp(out, min=0, max=1)


        return out, self.sigma, noized
