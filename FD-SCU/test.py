import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
import sys
from utils import Logger, display_config, set_random_seed

from torchsparse.utils.collate import sparse_collate
from data.face_scape_voxel_dataset import FaceScapeVoxelDataset
from sampling.voxelize import Voxelizer
from runner.voxel import valid
from models.model import FDSCU

parser = argparse.ArgumentParser()
# basic setting
parser.add_argument('--log_name', type=str, default='fdscu_test', help="log name")
parser.add_argument('--dataset', type=str ,default='face_scape_vox', help="dataset name")
parser.add_argument('--data_dir', type=str ,default="/home/hipeson/Wang_data/FaceScape/", help="dataset path")
parser.add_argument('--block', type=int, default=5, help='number of blocks in the model')
parser.add_argument('--channel', type=int, default=32, help='number of channels in the model')
parser.add_argument('--sampling', type=str, default="voxel", help="point sampling method")
parser.add_argument('--scale', type=int, default=2, help="input point cloud down-sampling scale")
parser.add_argument('--gt_scale', type=int, default=1, help="ground truth point cloud down-sampling scale")
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--log_path', type=str ,default='./result/log/', help="log path")
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--pretrain',
                    type=str,
                    default='./pretrained/FDSCU.pth',
                    help="log path")

parser.add_argument('--test_patch', type=int, default=1, help='number of patches sampled from an object')

args = parser.parse_args()


def main():
    sys.stdout = Logger(os.path.join(args.log_path, args.log_name + '.txt'))
    set_random_seed(args.seed)


    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')

    cudnn.benchmark = True
    display_config(args)
    start_time = time.time()

    # dataset setting
    print('Loading Dataset ...')

    if args.dataset == "face_scape_vox":
        test_set = FaceScapeVoxelDataset(args.data_dir, "test", gt_scale=1, num_patch=args.test_patch)
        test_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=1, shuffle=False,
                                 collate_fn=sparse_collate, pin_memory=True, drop_last=True)
    else:
        raise NotImplementedError

    if args.sampling == "voxel":
        sampler = Voxelizer(voxel_size=args.scale, avg_color=True)

    else:
        raise NotImplementedError
    bolck_size = 7
    model = FDSCU(voxel_size=args.scale, bolck_size=bolck_size, num_block=args.block,
                  num_channel=args.channel)

    model = model.cuda()
    model.load_state_dict(torch.load(args.pretrain))
    valid(test_loader, model, sampler, batch_size=1, verbose=args.verbose)


    finish_time = time.time()
    print("Time consumption:", finish_time - start_time)
    return


if __name__ == '__main__':
    main()