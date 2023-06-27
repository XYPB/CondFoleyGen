import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import sys
sys.path.append('..')
from models import r2plus1d18KeepTemp
from utils import torch_utils

class VideoOnsetNet(nn.Module):
    # Video Onset detection network
    def __init__(self, pretrained):
        super(VideoOnsetNet, self).__init__()
        self.net = r2plus1d18KeepTemp(pretrained=pretrained)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

    def forward(self, inputs, loss=False, evaluate=False):
        # import pdb; pdb.set_trace()
        x = inputs['frames']
        x = self.net(x)
        x = x.transpose(-1, -2)
        x = self.fc(x)
        x = x.squeeze(-1)

        return x


class BCLoss(nn.Module):
    # binary classification loss
    def __init__(self, args):
        super(BCLoss, self).__init__()

    def forward(self, pred, target):
        # import pdb; pdb.set_trace()
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        pos_weight = (target.shape[0] - target.sum()) / target.sum()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(pred.device)
        loss = criterion(pred, target.float())
        return loss

    def evaluate(self, pred, target):
        # import pdb; pdb.set_trace()

        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        pred = torch.sigmoid(pred)
        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()
        
        pos_index = np.nonzero(target == 1)[0]
        neg_index = np.nonzero(target == 0)[0]
        balance_num = min(pos_index.shape[0], neg_index.shape[0])
        index = np.concatenate((pos_index[:balance_num], neg_index[:balance_num]), axis=0)
        pred = pred[index]
        target = target[index]
        ap = average_precision_score(target, pred)
        acc = torch_utils.binary_acc(pred, target, thred=0.5)
        res = {
            'AP': ap,
            'Acc': acc
        }
        return res



if __name__ == '__main__':
    model = VideoOnsetNet(False).cuda()
    rand_input = torch.randn((1, 3, 30, 112, 112)).cuda()
    inputs = {
        'frames': rand_input
    }
    out = model(inputs)