import sys

import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, '.')  # nopep8
from specvqgan.modules.video_model.resnet import r2plus1d_18

FPS = 15

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class r2plus1d18KeepTemp(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        self.model = r2plus1d_18(pretrained=pretrained)

        self.model.layer2[0].conv1[0][3] = nn.Conv3d(230, 128, kernel_size=(3, 1, 1), 
            stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.model.layer2[0].downsample = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.model.layer3[0].conv1[0][3] = nn.Conv3d(460, 256, kernel_size=(3, 1, 1), 
            stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.model.layer3[0].downsample = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.model.layer4[0].conv1[0][3] = nn.Conv3d(921, 512, kernel_size=(3, 1, 1), 
            stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.model.layer4[0].downsample = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.model.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.model.fc = Identity()

        with torch.no_grad():
            rand_input = torch.randn((1, 3, 30, 112, 112))
            output = self.model(rand_input).detach().cpu()
            print('Validate Video feature shape: ', output.shape) # (1, 512, 30)

    def forward(self, x):
        N = x.shape[0]
        return self.model(x).reshape(N, 512, -1)

    def eval(self):
        return self
    
    def encode(self, c):
        info = None, None, c
        return c, None, info

    def decode(self, c):
        return c

    def get_input(self, batch, k, drop_cond=False):
        x = batch[k].cuda()
        x = x.permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format) # (N, 3, T, 112, 112)
        T = x.shape[2]
        if drop_cond:
            output = self.model(x) # (N, 512, T)
        else:
            cond_x = x[:, :, :T//2] # (N, 3, T//2, 112, 112)
            x = x[:, :, T//2:] # (N, 3, T//2, 112, 112)
            cond_feat = self.model(cond_x) # (N, 512, T//2)
            feat = self.model(x) # (N, 512, T//2)
            output = torch.cat([cond_feat, feat], dim=-1) # (N, 512, T)
        assert output.shape[2] == T
        return output


class resnet50(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Identity()
        # freeze resnet 50 model
        for params in self.model.parameters():
            params.requires_grad = False

    def forward(self, x):
        N = x.shape[0]
        return self.model(x).reshape(N, 2048)

    def eval(self):
        return self
    
    def encode(self, c):
        info = None, None, c
        return c, None, info

    def decode(self, c):
        return c

    def get_input(self, batch, k, drop_cond=False):
        x = batch[k].cuda()
        x = x.permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format) # (N, 3, T, 112, 112)
        T = x.shape[2]
        feats = []
        for t in range(T):
            xt = x[:, :, t]
            feats.append(self.model(xt))
        output = torch.stack(feats, dim=-1)
        assert output.shape[2] == T
        return output



if __name__ == '__main__':
    model = r2plus1d18KeepTemp(False).cuda()
    x = {'input': torch.randn((1, 60, 3, 112, 112))}
    out = model.get_input(x, 'input')
    print(out.shape)
