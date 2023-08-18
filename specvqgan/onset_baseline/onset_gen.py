import glob
import os
import numpy as np
from moviepy.editor import *
import librosa
import soundfile as sf

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
import torchvision.transforms as transforms
from PIL import Image
import shutil

from config import init_args
import data
import models
from models import *
from utils import utils, torch_utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


vision_transform_list = [
    transforms.Resize((128, 128)),
    transforms.CenterCrop((112, 112)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
video_transform = transforms.Compose(vision_transform_list)

def read_image(frame_list):
    imgs = []
    convert_tensor = transforms.ToTensor()
    for img_path in frame_list:
        image = Image.open(img_path).convert('RGB')
        image = convert_tensor(image)
        imgs.append(image.unsqueeze(0))
    # (T, C, H ,W)
    imgs = torch.cat(imgs, dim=0).squeeze()
    imgs = video_transform(imgs)
    imgs = imgs.permute(1, 0, 2, 3)
    # (C, T, H ,W)
    return imgs


def get_video_frames(origin_video_path):
    save_path = 'results/temp'
    if os.path.exists(save_path):
        os.system(f'rm -rf {save_path}')
    os.makedirs(save_path)
    command = f'ffmpeg -v quiet -y -i \"{origin_video_path}\" -f image2 -vf \"scale=-1:360,fps=15\" -qscale:v 3 \"{save_path}\"/frame%06d.jpg'
    os.system(command)
    frame_list = glob.glob(f'{save_path}/*.jpg')
    frame_list.sort()
    frame_list = frame_list[:2 * 15]
    frames = read_image(frame_list)
    return frames


def postprocess_video_onsets(probs, thres=0.5, nearest=5):
    # import pdb; pdb.set_trace()
    video_onsets = []
    pred = np.array(probs, copy=True)
    while True:
        max_ind = np.argmax(pred)
        video_onsets.append(max_ind)
        low = max(max_ind - nearest, 0)
        high = min(max_ind + nearest, pred.shape[0])
        pred[low: high] = 0
        if (pred > thres).sum() == 0:
            break
    video_onsets.sort()
    video_onsets = np.array(video_onsets)
    return video_onsets


def detect_onset_of_audio(audio, sample_rate):
    onsets = librosa.onset.onset_detect(
        y=audio, sr=sample_rate, units='samples', delta=0.3)
    return onsets


def get_onset_audio_range(audio_len, onsets, i):
    if i == 0:
        prev_offset = int(onsets[i] // 3)
    else:
        prev_offset = int((onsets[i] - onsets[i - 1]) // 3)

    if i == onsets.shape[0] - 1:
        post_offset = int((audio_len - onsets[i]) // 4 * 2)
    else:
        post_offset = int((onsets[i + 1] - onsets[i]) // 4 * 2)
    return prev_offset, post_offset


def generate_audio(con_videoclip, video_onsets):
    np.random.seed(2022)
    con_audioclip = con_videoclip.audio
    con_audio, con_sr = con_audioclip.to_soundarray(), con_audioclip.fps
    con_audio = con_audio.mean(-1)
    target_sr = 22050
    if target_sr != con_sr:
        con_audio = librosa.resample(con_audio, orig_sr=con_sr, target_sr=target_sr)
        con_sr = target_sr
    
    con_onsets = detect_onset_of_audio(con_audio, con_sr)
    gen_audio = np.zeros(int(2 * con_sr))

    for i in range(video_onsets.shape[0]):
        prev_offset, post_offset = get_onset_audio_range(
            int(con_sr * 2), video_onsets, i)
        j = np.random.choice(con_onsets.shape[0])
        prev_offset_con, post_offset_con = get_onset_audio_range(
            con_audio.shape[0], con_onsets, j)
        prev_offset = min(prev_offset, prev_offset_con)
        post_offset = min(post_offset, post_offset_con)
        gen_audio[video_onsets[i] - prev_offset: video_onsets[i] +
                post_offset] = con_audio[con_onsets[j] - prev_offset: con_onsets[j] + post_offset]
    return gen_audio


def generate_video(net, original_video_list, cond_video_list_0, cond_video_list_1, cond_video_list_2):
    save_folder = 'results/onset_baseline/vis'
    os.makedirs(save_folder, exist_ok=True)
    origin_video_folder = os.path.join(save_folder, '0_original')
    os.makedirs(origin_video_folder, exist_ok=True)

    for i in range(len(original_video_list)):
        # import pdb; pdb.set_trace()
        shutil.copy(original_video_list[i], os.path.join(
            origin_video_folder, original_video_list[i].split('/')[-1]))
        
        ori_videoclip = VideoFileClip(original_video_list[i])

        frames = get_video_frames(original_video_list[i])
        inputs = {
            'frames': frames.unsqueeze(0).to(device)
        }
        pred = net(inputs).squeeze()
        pred = torch.sigmoid(pred).data.cpu().numpy()
        video_onsets = postprocess_video_onsets(pred, thres=0.5, nearest=4)
        video_onsets = (video_onsets / 15 * 22050).astype(int)

        for ind, cond_video in enumerate([cond_video_list_0[i], cond_video_list_1[i], cond_video_list_2[i]]):
            cond_video_folder = os.path.join(save_folder, f'{ind * 2 + 1}_conditional_{ind}')
            os.makedirs(cond_video_folder, exist_ok=True)
            shutil.copy(cond_video, os.path.join(
                cond_video_folder, cond_video.split('/')[-1]))
            con_videoclip = VideoFileClip(cond_video)
            gen_audio = generate_audio(con_videoclip, video_onsets)
            save_audio_path = 'results/gen_audio.wav'
            sf.write(save_audio_path, gen_audio, 22050)
            gen_audioclip = AudioFileClip(save_audio_path)
            gen_videoclip = ori_videoclip.set_audio(gen_audioclip)
            save_gen_folder = os.path.join(save_folder, f'{ind * 2 + 2}_generate_{ind}')
            os.makedirs(save_gen_folder, exist_ok=True)
            gen_videoclip.write_videofile(os.path.join(save_gen_folder, original_video_list[i].split('/')[-1]))



if __name__ == '__main__': 
    net = models.VideoOnsetNet(pretrained=False).to(device)
    resume = 'checkpoints/EXP1/checkpoint_ep100.pth.tar'
    net, _ = torch_utils.load_model(resume, net, device=device, strict=True)
    read_folder = '' # name to a directory that generated with `audio_generation.py` 
    original_video_list = glob.glob(f'{read_folder}/2sec_full_orig_video/*.mp4')
    original_video_list.sort()

    cond_video_list_0 = glob.glob(f'{read_folder}/2sec_full_cond_video_0/*.mp4')
    cond_video_list_0.sort()

    cond_video_list_1 = glob.glob(f'{read_folder}/2sec_full_cond_video_1/*.mp4')
    cond_video_list_1.sort()

    cond_video_list_2 = glob.glob(f'{read_folder}/2sec_full_cond_video_2/*.mp4')
    cond_video_list_2.sort()

    generate_video(net, original_video_list, cond_video_list_0, cond_video_list_1, cond_video_list_2)