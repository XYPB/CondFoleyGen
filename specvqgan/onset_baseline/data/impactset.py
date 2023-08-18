from data import *
import pdb
from utils import sound, sourcesep
import csv
import glob
import h5py
import io
import json
import librosa
import numpy as np
import os
import pickle
from PIL import Image
from PIL import ImageFilter
import random
import scipy
import soundfile as sf
import time
from tqdm import tqdm
import glob
import cv2

import torch
import torch.nn as nn
import torchaudio
import torchvision.transforms as transforms
# import kornia as K
import sys
sys.path.append('..')


class CountixAVDataset(object):
    def __init__(self, args, split='train'):
        self.split = split
        if split == 'train':
            list_sample = './data/countixAV_train.json'
        elif split == 'val':
            list_sample = './data/countixAV_val.json'
        elif split == 'test':
            list_sample = './data/countixAV_test.json'

        # save args parameter
        self.repeat = args.repeat if split == 'train' else 1
        self.max_sample = args.max_sample

        self.video_transform = transforms.Compose(
            self.generate_video_transform(args))
        
        if isinstance(list_sample, str):
            with open(list_sample, "r") as f:
                self.list_sample = json.load(f)

        if self.max_sample > 0:
            self.list_sample = self.list_sample[0:self.max_sample]
        self.list_sample = self.list_sample * self.repeat

        random.seed(1234)
        np.random.seed(1234)
        num_sample = len(self.list_sample)
        if self.split == 'train':
            random.shuffle(self.list_sample)

        # self.class_dist = self.unbalanced_dist()
        print('Greatesthit Dataloader: # sample of {}: {}'.format(self.split, num_sample))


    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        info = self.list_sample[index]
        video_path = os.path.join('data', 'ImpactSet', 'impactset-proccess-resize', info)
        frame_path = os.path.join(video_path, 'frames')
        audio_path = os.path.join(video_path, 'audio')
        audio_path = glob.glob(f"{audio_path}/*_denoised.wav")[0]

        audio, audio_sample_rate = sf.read(audio_path, start=0, stop=1000, dtype='float64', always_2d=True)
        frame_rate = 15
        duration = 2.0
        frame_list = glob.glob(f'{frame_path}/*.jpg')
        frame_list.sort()

        frame_start = random.randint(0, len(frame_list))
        frame_start = min(frame_start, len(frame_list) - duration * frame_rate)
        frame_start = int(frame_start)
        
        frame_list = frame_list[frame_start: int(
            frame_start + np.ceil(duration * frame_rate))]
        audio_start = int(frame_start / frame_rate * audio_sample_rate)
        audio_end = int(audio_start + duration * audio_sample_rate)

        imgs = self.read_image(frame_list)
        audio, audio_rate = sf.read(audio_path, start=audio_start, stop=audio_end, dtype='float64', always_2d=True)
        audio = audio.mean(-1)

        onsets = librosa.onset.onset_detect(y=audio, sr=audio_rate, units='time', delta=0.3)
        onsets = np.rint(onsets * frame_rate).astype(int)
        onsets[onsets>29] = 29
        label = torch.zeros(len(frame_list))
        label[onsets] = 1

        batch = {
            'frames': imgs,
            'label': label
        }
        return batch

    def getitem_test(self, index):
        self.__getitem__(index)

    def __len__(self):
        return len(self.list_sample)


    def read_image(self, frame_list):
        imgs = []
        convert_tensor = transforms.ToTensor()
        for img_path in frame_list:
            image = Image.open(img_path).convert('RGB')
            image = convert_tensor(image)
            imgs.append(image.unsqueeze(0))
        # (T, C, H ,W)
        imgs = torch.cat(imgs, dim=0).squeeze()
        imgs = self.video_transform(imgs)
        imgs = imgs.permute(1, 0, 2, 3)
        # (C, T, H ,W)
        return imgs

    def generate_video_transform(self, args):
        resize_funct = transforms.Resize((128, 128))
        if self.split == 'train':
            crop_funct = transforms.RandomCrop(
                (112, 112))
            color_funct = transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0, hue=0)
        else:
            crop_funct = transforms.CenterCrop(
                (112, 112))
            color_funct = transforms.Lambda(lambda img: img)

        vision_transform_list = [
            resize_funct,
            crop_funct,
            color_funct,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return vision_transform_list