import argparse
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import init_args
import data
import models
from models import *
from utils import utils, torch_utils


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validation(args, net, criterion, data_loader, device='cuda'):
    # import pdb; pdb.set_trace()
    net.eval()
    pred_all = torch.tensor([]).to(device)
    target_all = torch.tensor([]).to(device)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Validation"):
            pred, target = predict(args, net, batch, device)
            pred_all = torch.cat([pred_all, pred], dim=0)
            target_all = torch.cat([target_all, target], dim=0)

    res = criterion.evaluate(pred_all, target_all)
    torch.cuda.empty_cache()
    net.train()
    return res


def predict(args, net, batch, device):
    inputs = {
        'frames': batch['frames'].to(device)
    }
    pred = net(inputs)   
    target = batch['label'].to(device)  
    return pred, target


def train(args, device):
    # save dir
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    # ----- make dirs for checkpoints ----- #
    sys.stdout = utils.LoggerOutput(os.path.join('checkpoints', args.exp, 'log.txt'))
    os.makedirs('./checkpoints/' + args.exp, exist_ok=True)

    writer = SummaryWriter(os.path.join('./checkpoints', args.exp, 'visualization'))
    # ------------------------------------- #
    tqdm.write('{}'.format(args)) 

    # ------------------------------------ #

    
    # ----- Dataset and Dataloader ----- #
    train_dataset = data.GreatestHitDataset(args, split='train')
    # train_dataset.getitem_test(1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)
    
    val_dataset = data.GreatestHitDataset(args, split='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)
    # --------------------------------- #

    # ----- Network ----- #
    net = models.VideoOnsetNet(pretrained=False).to(device)
    criterion = models.BCLoss(args)
    optimizer = torch_utils.make_optimizer(net, args)
    # --------------------- #

    # -------- Loading checkpoints weights ------------- #
    if args.resume:
        resume = './checkpoints/' + args.resume
        net, args.start_epoch = torch_utils.load_model(resume, net, device=device, strict=True)
        if args.resume_optim:
            tqdm.write('loading optimizer...')
            optim_state = torch.load(resume)['optimizer']
            optimizer.load_state_dict(optim_state)
            tqdm.write('loaded optimizer!')
        else:
            args.start_epoch = 0

    # ------------------- 
    net = nn.DataParallel(net, device_ids=gpu_ids)
    #  --------- Random or resume validation ------------ #
    res = validation(args, net, criterion, val_loader, device)
    writer.add_scalars('VideoOnset' + '/validation', res, args.start_epoch)
    tqdm.write("Beginning, Validation results: {}".format(res))
    tqdm.write('\n')

    # ----------------- Training ---------------- #
    # import pdb; pdb.set_trace()
    VALID_STEP = args.valid_step
    for epoch in range(args.start_epoch, args.epochs):
        running_loss = 0.0
        torch_utils.adjust_learning_rate(optimizer, epoch, args)
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            pred, target = predict(args, net, batch, device)
            loss = criterion(pred, target)        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1 == 0:
                tqdm.write("Epoch: {}/{}, step: {}/{}, loss: {}".format(epoch+1, args.epochs, step+1, len(train_loader), loss))
                running_loss += loss.item()

            current_step = epoch * len(train_loader) + step + 1
            BOARD_STEP = 3
            if (step+1) % BOARD_STEP == 0:
                writer.add_scalar('VideoOnset' + '/training loss', running_loss / BOARD_STEP, current_step)
                running_loss = 0.0
        
        
        # ----------- Validtion -------------- #
        if (epoch + 1) % VALID_STEP == 0:
            res = validation(args, net, criterion, val_loader, device)
            writer.add_scalars('VideoOnset' + '/validation', res, epoch + 1)
            tqdm.write("Epoch: {}/{}, Validation results: {}".format(epoch + 1, args.epochs, res))

        # ---------- Save model ----------- #
        SAVE_STEP = args.save_step
        if (epoch + 1) % SAVE_STEP == 0:
            path = os.path.join('./checkpoints', args.exp, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar')
            torch.save({'epoch': epoch + 1,
                        'step': current_step,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        },
                        path)
        # --------------------------------- #
    torch.cuda.empty_cache()
    tqdm.write('Training Complete!')
    writer.close()


def test(args, device):
    # save dir
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    # ----- make dirs for results ----- #
    sys.stdout = utils.LoggerOutput(os.path.join('results', args.exp, 'log.txt'))
    os.makedirs('./results/' + args.exp, exist_ok=True)
    # ------------------------------------- #
    tqdm.write('{}'.format(args)) 
    # ------------------------------------ #
    # ----- Dataset and Dataloader ----- #
    test_dataset = data.GreatestHitDataset(args, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)

    # --------------------------------- #
    # ----- Network ----- #
    net = models.VideoOnsetNet(pretrained=False).to(device)
    criterion = models.BCLoss(args)
    # -------- Loading checkpoints weights ------------- #
    if args.resume:
        resume = './checkpoints/' + args.resume
        net, _ = torch_utils.load_model(resume, net, device=device, strict=True)

    # ------------------- #
    net = nn.DataParallel(net, device_ids=gpu_ids)
    #  --------- Testing ------------ #
    res = validation(args, net, criterion, test_loader, device)
    tqdm.write("Testing results: {}".format(res))


# CUDA_VISIBLE_DEVICES=1 python main.py --exp='EXP1' --epochs=100 --batch_size=12 --num_workers=8 --save_step=10 --valid_step=1 --lr=0.0001 --optim='Adam' --repeat=1 --schedule='cos'
if __name__ == '__main__':
    args = init_args()
    if args.test_mode:
        test(args, DEVICE)
    else:
        train(args, DEVICE)