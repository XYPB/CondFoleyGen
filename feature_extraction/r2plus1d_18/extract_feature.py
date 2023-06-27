import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
import torchvision
import random
import os
import json
import tqdm

from .data import GreatestHit, get_transform3D, non_negative

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='data/greatesthit/greatesthit_processed')

parser.add_argument('-b', '--batch_size', default=48, type=int)
parser.add_argument('-n', '--worker', default=16, type=int)
parser.add_argument('--gap', default=0, type=float)
parser.add_argument('--fps', default=15, type=float)

SAVE_DIR = 'data/greatesthit/feature_r2plus1d_dim1024_15fps'
SR = 22050

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    """A shape adaptation layer to patch certain networks."""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Unsqueeze(nn.Module):
    """A shape adaptation layer to patch certain networks."""

    def __init__(self):
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        return x.unsqueeze(-1)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def random_weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def get_video_feature_extractor(vid_base_arch='r2plus1d_18', pretrained=False, duration=1):
    if vid_base_arch == 'r2plus1d_18':
        model = torchvision.models.video.__dict__[
            vid_base_arch](pretrained=pretrained)
        if not pretrained:
            print("Randomy initializing models")
            random_weight_init(model)
        model.fc = Identity()
    elif vid_base_arch == 'r3d_9':
        print('r3d_9')
        model = torchvision.models.video.resnet._video_resnet('r3d_18',
                                                              pretrained=False, progress=False,
                                                              block=torchvision.models.video.resnet.BasicBlock,
                                                              conv_makers=[
                                                                  torchvision.models.video.resnet.Conv3DSimple] * 4,
                                                              layers=[
                                                                  1, 1, 1, 1],
                                                              stem=torchvision.models.video.resnet.BasicStem)
        if not pretrained:
            print("Random initializing models")
            random_weight_init(model)
        model.fc = Identity()
    elif vid_base_arch == 'resnet50':
        model = torchvision.models.resnet50(True)
        model.fc = Identity()
    return model


if __name__ == '__main__':
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    model = get_video_feature_extractor('resnet50', True)
    # model.avgpool = nn.AdaptiveAvgPool3d((2, 1, 1))
    print(model)
    print(f'==> Training with {torch.cuda.device_count()} GPUs')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()

    for param in model.parameters():
        param.requires_grad = False

    print('==> Preparing data and model...')
    ######## TRANSFORM ########
    vis_transform, vis_transform_vate, aud_transform = get_transform3D()

    video_list = sorted(os.listdir(args.root))
    ghit = GreatestHit(args.root, video_list, frame_transform=vis_transform_vate, conditional=True, return_frame_path=False,
                       n_negative=0, length=1., gap=args.gap, fps=15, iter_all_hit=True, drop_none=False)
    ghit_loader = DataLoader(ghit, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.worker, drop_last=False)

    model.eval()
    pbar = tqdm.tqdm(enumerate(ghit_loader), total=len(ghit_loader))
    V_FEAT = []
    V_INFO = {'video_name': [], 'start_idx':[], 'hit_type': []}
    with torch.no_grad():
        for i, (X_frame, tar_type) in pbar:
            v, ct, hit_type = tar_type
            X = X_frame.float().cuda()
            video_feat = model(X).detach().cpu()

            V_FEAT.append(video_feat)

            V_INFO['video_name'] += [str(vv) for vv in v]
            V_INFO['start_idx'] += [int((float(t) - 0.5) * SR) for t in ct]
            V_INFO['hit_type'] += [str(t) for t in hit_type]
    V_FEAT = torch.cat(V_FEAT, dim=0)
    torch.save(V_FEAT, os.path.join(SAVE_DIR, 'feature_r2plus1d_dim1024_15fps.pkl'))
    json.dump(V_INFO, open(os.path.join(SAVE_DIR, 'info_r2plus1d_dim1024_15fps.json'), 'w'))

