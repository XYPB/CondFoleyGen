'''
The code is partially borrowed from:
https://github.com/v-iashin/video_features/blob/861efaa4ed67/utils/utils.py
and
https://github.com/PeihaoChen/regnet/blob/199609/extract_audio_and_video.py
'''
import os
from pyexpat import model
import shutil
import subprocess
from glob import glob
from pathlib import Path
from typing import Dict
import json
import noisereduce as nr
import copy
from random import sample

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from omegaconf.omegaconf import OmegaConf
from sample_visualization import (load_feature_extractor,
                                  load_model_from_config, load_vocoder)
from specvqgan.data.greatesthit import CropFeats
from specvqgan.data.greatesthit import get_GH_data_identifier, non_negative
from specvqgan.util import download, md5_hash
from specvqgan.models.cond_transformer import disabled_train
from train import instantiate_from_config
from specvqgan.data.transforms import *

from feature_extraction.extract_mel_spectrogram_greatesthit import get_spectrogram_infer
from feature_extraction.extract_mel_spectrogram import get_spectrogram as get_spectrogram_non_normalize

plt.rcParams['savefig.bbox'] = 'tight'

SR = 22050
FPS = 15


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library

    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path

def which_ffprobe() -> str:
    '''Determines the path to ffprobe library

    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffprobe'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffprobe_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffprobe_path


def check_video_for_audio(path):
    assert which_ffprobe() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    cmd = f'{which_ffprobe()} -loglevel error -show_entries stream=codec_type -of default=nw=1 {path}'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = result.stdout.decode('utf-8')
    print(result)
    return 'codec_type=audio' in result

def get_duration(path):
    assert which_ffprobe() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    cmd = f'{which_ffprobe()} -hide_banner -loglevel panic' \
          f' -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {path}'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    duration = float(result.stdout.decode('utf-8').replace('\n', ''))
    return duration

def trim_video(video_path: str, start: int, trim_duration: int = 10, tmp_path: str = './tmp', cond: bool = False):
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    if Path(video_path).suffix != '.mp4':
        print(f'File Extension is not `mp4` (it is {Path(video_path).suffix}). It will be re-encoded to mp4.')

    video_duration = get_duration(video_path)
    print('Video Duration:', video_duration)
    assert video_duration > start, f'Video Duration < Trim Start: {video_duration} < {start}'

    # create tmp dir if doesn't exist
    os.makedirs(tmp_path, exist_ok=True)
    if not cond:
        trim_vid_path = os.path.join(tmp_path, f'{Path(video_path).stem}_trim_to_{int(trim_duration)}s_from_{start:.4f}.mp4')
    else:
        trim_vid_path = os.path.join(tmp_path, f'{Path(video_path).stem}_cond_trim_to_{int(trim_duration)}s_from_{start:.4f}.mp4')
    cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic' \
          f' -i {video_path} -ss {start} -t {int(trim_duration)} -y {trim_vid_path}'
    subprocess.call(cmd.split())
    print('Trimmed the input video', video_path, 'and saved the output @', trim_vid_path)

    return trim_vid_path


def resize_video(video_path: str, new_H: int, new_W: int):
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    if Path(video_path).suffix != '.mp4':
        print(f'File Extension is not `mp4` (it is {Path(video_path).suffix}). It will be re-encoded to mp4.')
    dest = video_path.replace('.mp4', '_resized.mp4')
    cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic -i {video_path} -vf scale={new_W}:{new_H} -y {dest}'
    subprocess.call(cmd.split())
    return dest


def reencode_video_with_diff_fps(video_path: str, tmp_path: str, extraction_fps: int) -> str:
    '''Reencodes the video given the path and saves it to the tmp_path folder.

    Args:
        video_path (str): original video
        tmp_path (str): the folder where tmp files are stored (will be appended with a proper filename).
        extraction_fps (int): target fps value

    Returns:
        str: The path where the tmp file is stored. To be used to load the video from
    '''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    # assert video_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'
    # create tmp dir if doesn't exist
    os.makedirs(tmp_path, exist_ok=True)

    # form the path to tmp directory
    new_path = os.path.join(tmp_path, f'{Path(video_path).stem}_new_fps.mp4')
    cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic '
    cmd += f'-y -i {video_path} -filter:v fps=fps={extraction_fps} {new_path}'
    subprocess.call(cmd.split())

    return new_path

def maybe_download_model(model_name: str, log_dir: str) -> str:
    name2info = {
        '2021-06-20T16-35-20_vggsound_transformer': {
            'info': 'No Feats',
            'hash': 'b1f9bb63d831611479249031a1203371',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-06-20T16-35-20_vggsound_transformer.tar.gz',
        },
        '2021-07-30T21-03-22_vggsound_transformer': {
            'info': '1 ResNet50 Feature',
            'hash': '27a61d4b74a72578d13579333ed056f6',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-07-30T21-03-22_vggsound_transformer.tar.gz',
        },
        '2021-07-30T21-34-25_vggsound_transformer': {
            'info': '5 ResNet50 Features',
            'hash': 'f4d7105811589d441b69f00d7d0b8dc8',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-07-30T21-34-25_vggsound_transformer.tar.gz',
        },
        '2021-07-30T21-34-41_vggsound_transformer': {
            'info': '212 ResNet50 Features',
            'hash': 'b222cc0e7aeb419f533d5806a08669fe',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-07-30T21-34-41_vggsound_transformer.tar.gz',
        },
        '2021-06-03T00-43-28_vggsound_transformer': {
            'info': 'Class Label',
            'hash': '98a3788ab973f1c3cc02e2e41ad253bc',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-06-03T00-43-28_vggsound_transformer.tar.gz',
        },
        '2021-05-19T22-16-54_vggsound_codebook': {
            'info': 'VGGSound Codebook',
            'hash': '7ea229427297b5d220fb1c80db32dbc5',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-05-19T22-16-54_vggsound_codebook.tar.gz',
        }
    }
    if model_name not in name2info.keys():
        return os.path.join(log_dir, model_name)
    print(f'Using: {model_name} ({name2info[model_name]["info"]})')
    model_dir = os.path.join(log_dir, model_name)
    if not os.path.exists(model_dir):
        tar_local_path = os.path.join(log_dir, f'{model_name}.tar.gz')
        # check if tar already exists and its md5sum
        if not os.path.exists(tar_local_path) or md5_hash(tar_local_path) != name2info[model_name]['hash']:
            down_link = name2info[model_name]['link']
            download(down_link, tar_local_path)
            print('Unpacking', tar_local_path, 'to', log_dir)
            shutil.unpack_archive(tar_local_path, log_dir)
            # clean-up space as we already have unpacked folder
            os.remove(tar_local_path)
    return model_dir

def load_config(model_dir: str):
    # Load the config
    config_main = sorted(glob(os.path.join(model_dir, 'configs/*-project.yaml')))[-1]
    config_pylt = sorted(glob(os.path.join(model_dir, 'configs/*-lightning.yaml')))[-1]
    config = OmegaConf.merge(
        OmegaConf.load(config_main),
        OmegaConf.load(config_pylt),
    )
    # patch config. E.g. if the model is trained on another machine with different paths
    for a in ['spec_dir_path', 'rgb_feats_dir_path', 'flow_feats_dir_path']:
        if config.data.params[a] is not None:
            if 'vggsound.VGGSound' in config.data.params.train.target:
                base_path = './data/vggsound/'
            elif 'vas.VAS' in config.data.params.train.target:
                base_path = './data/vas/features/*/'
            elif 'greatesthit.GreatestHit' in config.data.params.train.target:
                base_path = './data/greatesthit/'
            else:
                raise NotImplementedError
            config.data.params[a] = os.path.join(base_path, Path(config.data.params[a]).name)
    return config

def load_model(model_name, log_dir, device, load_new_first_stage=False):
    to_use_gpu = True if device.type == 'cuda' else False
    model_dir = maybe_download_model(model_name, log_dir)
    config = load_config(model_dir)

    # Sampling model
    ckpt = sorted(glob(os.path.join(model_dir, 'checkpoints/*.ckpt')))[-1]
    pl_sd = torch.load(ckpt, map_location='cpu')
    sampler = load_model_from_config(config.model, pl_sd['state_dict'], to_use_gpu, load_new_first_stage=load_new_first_stage)['model']
    sampler.to(device)

    # aux models (vocoder and melception)
    ckpt_melgan = config.lightning.callbacks.image_logger.params.vocoder_cfg.params.ckpt_vocoder
    melgan = load_vocoder(ckpt_melgan, eval_mode=True)['model'].to(device)
    melception = load_feature_extractor(to_use_gpu, eval_mode=True)
    return config, sampler, melgan, melception

def load_neural_audio_codec(model_name, log_dir, device):
    model_dir = maybe_download_model(model_name, log_dir)
    config = load_config(model_dir)

    config.model.params.ckpt_path = f'./logs/{model_name}/checkpoints/last.ckpt'
    if not os.path.exists(config.model.params.ckpt_path):
        ckpt_list = glob(f'./logs/{model_name}/checkpoints/*.ckpt')
        ckpt_list = sorted(ckpt_list)
        config.model.params.ckpt_path = ckpt_list[-1]
    print(config.model.params.ckpt_path)
    model = instantiate_from_config(config.model)
    model = model.to(device)
    model = model.eval()
    model.train = disabled_train
    vocoder = load_vocoder(Path('./vocoder/logs/vggsound/'), eval_mode=True)['model'].to(device)
    return config, model, vocoder

class LeftmostCropOrTile(object):
    def __init__(self, crop_or_tile_to):
        self.crop_or_tile_to = crop_or_tile_to

    def __call__(self, item: Dict):
        # tile or crop features to the `crop_or_tile_to`
        T, D = item['feature'].shape
        if T != self.crop_or_tile_to:
            how_many_tiles_needed = 1 + (self.crop_or_tile_to // T)
            item['feature'] = np.tile(item['feature'], (how_many_tiles_needed, 1))[:self.crop_or_tile_to, :]
        return item

class ExtractResNet50(torch.nn.Module):

    def __init__(self, extraction_fps, feat_cfg, device, batch_size=32, tmp_dir='./tmp'):
        super(ExtractResNet50, self).__init__()
        self.tmp_path = tmp_dir
        self.extraction_fps = extraction_fps
        self.batch_size = batch_size
        self.feat_cfg = feat_cfg

        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.means, std=self.stds)
        ])
        random_crop = False
        self.post_transforms = transforms.Compose([
            LeftmostCropOrTile(feat_cfg.feat_len),
            CropFeats([feat_cfg.feat_crop_len, feat_cfg.feat_depth], random_crop),
            (lambda x: x) if feat_cfg.feat_sampler_cfg is None else instantiate_from_config(feat_cfg.feat_sampler_cfg),
        ])
        self.device = device
        self.model = models.resnet50(pretrained=True).to(device)
        self.model.eval()
        # save the pre-trained classifier for show_preds and replace it in the net with identity
        self.model_class = self.model.fc
        self.model.fc = torch.nn.Identity()

    @torch.no_grad()
    def forward(self, video_path: str) -> Dict[str, np.ndarray]:

        if self.feat_cfg.replace_feats_with_random:
            T, D = self.feat_cfg.feat_sampler_cfg.params.feat_sample_size, self.feat_cfg.feat_depth
            print(f'Since we are in "No Feats" setting, returning a random feature: [{T}, {D}]')
            random_features = {'feature': torch.rand(T, D)}
            return random_features, []

        # take the video, change fps and save to the tmp folder
        if self.extraction_fps is not None:
            video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

        # read a video
        cap = cv2.VideoCapture(video_path)
        batch_list = []
        vid_feats = []
        cached_frames = []
        transforms_for_show = transforms.Compose(self.transforms.transforms[:4])
        # sometimes when the target fps is 1 or 2, the first frame of the reencoded video is missing
        # and cap.read returns None but the rest of the frames are ok. timestep is 0.0 for the 2nd frame in
        # this case
        first_frame = True

        # iterating through the opened video frame-by-frame and occationally run the model once a batch is
        # formed
        while cap.isOpened():
            frame_exists, rgb = cap.read()

            if first_frame and not frame_exists:
                continue
            first_frame = False

            if frame_exists:
                # prepare data and cache if needed
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                cached_frames.append(transforms_for_show(rgb))
                rgb = self.transforms(rgb).unsqueeze(0).to(self.device)
                batch_list.append(rgb)
                # when batch is formed to inference
                if len(batch_list) == self.batch_size:
                    batch_feats = self.model(torch.cat(batch_list))
                    vid_feats.extend(batch_feats.tolist())
                    # clean up the batch list
                    batch_list = []
            else:
                # if the last batch was smaller than the batch size, we still need to process those frames
                if len(batch_list) != 0:
                    batch_feats = self.model(torch.cat(batch_list))
                    vid_feats.extend(batch_feats.tolist())
                cap.release()
                break

        vid_feats = np.array(vid_feats)
        features = {'feature': vid_feats}
        print('Raw Extracted Representation:', features['feature'].shape)

        if self.post_transforms is not None:
            features = self.post_transforms(features)
            # using 'feature' as the key to reuse the feature resampling transform
            cached_frames = self.post_transforms.transforms[-1]({'feature': torch.stack(cached_frames)})['feature']

        print('Post-processed Representation:', features['feature'].shape)

        return features, cached_frames


class LoadR2plus1DFrames(object):

    def __init__(self, feat_depth, feat_crop_len, random_crop, L=1.0,
                meta_path='./data/info_r2plus1d_dim1024_15fps.json', 
                frame_path='data/greatesthit/greatesthit_processed'):
        self.frame_transforms = transforms.Compose([
            Resize3D(128),
            CenterCrop3D(112),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.feats_transforms = CropFeats([feat_crop_len, feat_depth], random_crop)
        self.meta_path = meta_path
        self.frame_path = frame_path
        self.feat_depth = feat_depth
        self.feat_crop_len = feat_crop_len
        self.L = L

        greatesthit_meta = json.load(open(self.meta_path, 'r'))
        unique_classes = sorted(list(set(ht for ht in greatesthit_meta['hit_type'])))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}
        self.target2label = {target: label for label, target in self.label2target.items()}
        self.video_idx2label = {
            get_GH_data_identifier(greatesthit_meta['video_name'][i], greatesthit_meta['start_idx'][i]): 
            greatesthit_meta['hit_type'][i] for i in range(len(greatesthit_meta['video_name']))
        }
        self.available_video_hit = list(self.video_idx2label.keys())
        self.video_idx2idx = {
            get_GH_data_identifier(greatesthit_meta['video_name'][i], greatesthit_meta['start_idx'][i]):
            i for i in range(len(greatesthit_meta['video_name']))
        }

        self.dataset = self.available_video_hit

        self.video2indexes = {}
        for video_idx in self.dataset:
            video, start_idx = video_idx.split('_')
            if video not in self.video2indexes.keys():
                self.video2indexes[video] = []
            self.video2indexes[video].append(start_idx)
        for video in self.video2indexes.keys():
            if len(self.video2indexes[video]) == 1: # given video contains only one hit
                self.dataset.remove(
                    get_GH_data_identifier(video, self.video2indexes[video][0])
                )
        
        if L != 1.0:
            self.validate_data()

    def validate_data(self):
        original_len = len(self.dataset)
        valid_dataset = []
        for video_idx in self.dataset:
            video, start_idx = video_idx.split('_')
            frame_path = os.path.join(self.frame_path, video, 'frames')
            start_frame_idx = non_negative(FPS * int(start_idx)/SR)
            end_frame_idx = non_negative(start_frame_idx + FPS * (self.L + 0.6))
            if os.path.exists(os.path.join(frame_path, f'frame{end_frame_idx:0>6d}.jpg')):
                valid_dataset.append(video_idx)
            else:
                self.video2indexes[video].remove(start_idx)
        for video_idx in valid_dataset:
            video, start_idx = video_idx.split('_')
            if len(self.video2indexes[video]) == 1:
                valid_dataset.remove(video_idx)
        if original_len != len(valid_dataset):
            print(f'Validated dataset with enough frames: {len(valid_dataset)}')
        self.dataset = valid_dataset

    def sample2Clips(self, video_path, shift=0, cond_shift=0):
        item = {}

        video_path = reencode_video_with_diff_fps(video_path, './tmp', FPS)

        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = list(range(1, length - int((self.L + max(shift, cond_shift)) * FPS)))
        start_frame_idx = sample(frame_idx, k=1)[0] + int(shift * FPS)
        tar_frame_idx = [i for i in range(start_frame_idx, start_frame_idx + int(self.L * FPS))]
        for idx in tar_frame_idx:
            if idx in frame_idx:
                frame_idx.remove(idx)

        cond_start_frame_idx = sample(frame_idx, k=1)[0] + int(cond_shift * FPS)
        cond_frame_idx = [i for i in range(cond_start_frame_idx, cond_start_frame_idx + int(self.L * FPS))]

        frames = []
        cond_frames = []
        first_frame = True
        idx = -1
        while cap.isOpened():
            idx += 1
            frame_exists, rgb = cap.read()

            if first_frame and not frame_exists:
                continue
            first_frame = False

            if frame_exists:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = Image.fromarray(np.array(rgb))
                if idx in tar_frame_idx:
                    frames.append(rgb)
                elif idx in cond_frame_idx:
                    cond_frames.append(rgb)
            else:
                cap.release()
                break

        if self.frame_transforms is not None:
            cond_frames = self.frame_transforms(cond_frames)
            frames = self.frame_transforms(frames)
        

        start_idx = non_negative(SR * float(start_frame_idx) / FPS)
        cond_start_idx = non_negative(SR * float(cond_start_frame_idx) / FPS)

        item['feature'] = np.stack(cond_frames + frames, axis=0) # (30, 3, 112, 112)
        item['file_path_feats_'] = (video_path, start_frame_idx)
        item['file_path_cond_feats_'] = (video_path, cond_start_frame_idx)
        item['idx'] = (cond_start_idx, start_idx)

        item['label'] = 'outside'
        item['target'] = 'outside'

        return item

    def __call__(self, video_path, shift=0, cond_shift=0, load_frame=True):
        item = {}

        video = video_path.split('/')[-1].split('_')[0]
        if video not in self.video2indexes.keys():
            return self.sample2Clips(video_path, shift, cond_shift)
        video_hits_idx = copy.copy(self.video2indexes[video])
        idxes = sample(video_hits_idx, k=2)
        cond_start_idx = min(idxes)
        start_idx = max(idxes)

        video_idx = get_GH_data_identifier(video, start_idx)

        video, start_idx = video_idx.split('_')
        frame_path = os.path.join(self.frame_path, video, 'frames')
        start_frame_idx = non_negative(FPS * int(start_idx)/SR)
        end_frame_idx = non_negative(start_frame_idx + int(FPS * self.L))

        start_frame_idx += int(FPS * shift)
        end_frame_idx += int(FPS * shift)

        if load_frame:
            frames = [Image.open(os.path.join(
                frame_path, f'frame{i+1:0>6d}.jpg')).convert('RGB') for i in 
                range(start_frame_idx, end_frame_idx)]
        else:
            frames = []

        cond_video_idx = get_GH_data_identifier(video, cond_start_idx)

        cond_video, cond_start_idx = cond_video_idx.split('_')
        cond_frame_path = os.path.join(self.frame_path, cond_video, 'frames')
        cond_start_frame_idx = non_negative(FPS * int(cond_start_idx)/SR)
        cond_end_frame_idx = non_negative(cond_start_frame_idx + int(FPS * self.L))

        cond_start_frame_idx += int(FPS * cond_shift)
        cond_end_frame_idx += int(FPS * cond_shift)

        if load_frame:
            cond_frames = [Image.open(os.path.join(
                cond_frame_path, f'frame{i+1:0>6d}.jpg')).convert('RGB') for i in 
                range(cond_start_frame_idx, cond_end_frame_idx)]
        else:
            cond_frames = []

        if self.frame_transforms is not None and load_frame:
            cond_frames = self.frame_transforms(cond_frames)
            frames = self.frame_transforms(frames)

        if load_frame:
            item['feature'] = np.stack(cond_frames + frames, axis=0) # (30 * T, 3, 112, 112)
        item['file_path_feats_'] = (frame_path, start_frame_idx)
        item['file_path_cond_feats_'] = (cond_frame_path, cond_start_frame_idx)
        item['idx'] = (cond_start_idx, start_idx)

        item['label'] = self.video_idx2label[video_idx]
        item['target'] = self.label2target[item['label']]

        return item


class LoadR2plus1DFeat(object):

    def __init__(self, feat_depth, feat_crop_len, random_crop, sr=22050, meta_path='./data/info_r2plus1d_dim1024_15fps.json',
                feats_path='data/greatesthit/feature_r2plus1d_dim1024_15fps/feature_r2plus1d_dim1024_15fps.pkl') -> None:
        self.meta_path = meta_path
        self.sr = sr
        self.feats_transforms = CropFeats([feat_crop_len, feat_depth], random_crop)

        greatesthit_meta = json.load(open(self.meta_path, 'r'))
        unique_classes = sorted(list(set(ht for ht in greatesthit_meta['hit_type'])))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}
        self.target2label = {target: label for label, target in self.label2target.items()}
        self.video_idx2label = {
            get_GH_data_identifier(greatesthit_meta['video_name'][i], greatesthit_meta['start_idx'][i]): 
            greatesthit_meta['hit_type'][i] for i in range(len(greatesthit_meta['video_name']))
        }
        self.available_video_hit = list(self.video_idx2label.keys())
        self.video_idx2idx = {
            get_GH_data_identifier(greatesthit_meta['video_name'][i], greatesthit_meta['start_idx'][i]):
            i for i in range(len(greatesthit_meta['video_name']))
        }
        self.video2indexes = {}
        self.dataset = self.available_video_hit
        for video_idx in self.dataset:
            video, start_idx = video_idx.split('_')
            if video not in self.video2indexes.keys():
                self.video2indexes[video] = []
            self.video2indexes[video].append(start_idx)
        for video in self.video2indexes.keys():
            if len(self.video2indexes[video]) == 1: # given video contains only one hit
                self.dataset.remove(
                    get_GH_data_identifier(video, self.video2indexes[video][0])
                )

        self.feats = torch.load(feats_path).numpy()

    def __call__(self, video_path):
        video_name = video_path.split('/')[-1].split('_')[0]

        indexes = self.video2indexes[video_name]
        item = {}
        for idx in indexes:
            video_idx = get_GH_data_identifier(video_name, idx)
            feat_idx = self.video_idx2idx[video_idx]
            feat = self.feats[feat_idx]
            video_start_t = float(idx) / float(self.sr)
            label = self.video_idx2label[video_idx] # str
            target = self.label2target[label] # int
            item[idx] = {
                'feature': feat,
                'video_start_t': video_start_t,
                'hit_type': label,
                'target': target
            }
        return item


def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    cached_frames = []
    first_frame = True

    while cap.isOpened():
        frame_exists, rgb = cap.read()

        if first_frame and not frame_exists:
            continue
        first_frame = False

        if frame_exists:
            # prepare data and cache if needed
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            cached_frames.append(rgb)
        else:
            break

    return cached_frames


def extract_melspectrogram(in_path: str, sr: int, duration: int = 10, tmp_path: str = './tmp', normalize: bool=False, remove_noise: bool=False, using_torch: bool=False, return_wav=False) -> np.ndarray:
    '''Extract Melspectrogram similar to RegNet.'''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    # assert in_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'
    # create tmp dir if doesn't exist
    os.makedirs(tmp_path, exist_ok=True)

    # Extract audio from a video if needed
    if in_path.endswith('.wav'):
        audio_raw = in_path
    else:
        audio_raw = os.path.join(tmp_path, f'{Path(in_path).stem}.wav')
        cmd = f'{which_ffmpeg()} -i {in_path} -hide_banner -loglevel panic -f wav -vn -y {audio_raw}'
        subprocess.call(cmd.split())

    # Extract audio from a video
    audio_new = os.path.join(tmp_path, f'{Path(in_path).stem}_{sr}hz.wav')
    cmd = f'{which_ffmpeg()} -i {audio_raw} -hide_banner -loglevel panic -ac 1 -ab 16k -ar {sr} -y {audio_new}'
    subprocess.call(cmd.split())

    if remove_noise:
        audio_denoised = audio_new.replace('.wav', '_denoised.wav')
        # cmd = f'{which_ffmpeg()} -i {audio_new} -hide_banner -loglevel panic -ac 1 -ab 16k -ar {sr} -acodec pcm_s16le -af bandpass=frequency=1470:width_type=h:width=2940 -y {audio_denoised}'
        # print(cmd)
        wav, sr = soundfile.read(audio_new)
        if len(wav.shape) == 1:
            wav = wav[None, :]
        wav_rn = nr.reduce_noise(y=wav, sr=sr, n_fft=1024, hop_length=1024//4)
        wav_rn = wav_rn.squeeze()
        soundfile.write(audio_denoised, wav_rn, samplerate=sr)
        audio_new = audio_denoised
        # subprocess.call(cmd.split())
        # audio_new = audio_denoised

    length = int(duration * sr)
    if using_torch:
        audio_zero_pad, spec = get_spectrogram_torch(audio_new, save_dir=None, length=length, save_results=False)
    elif normalize:
        audio_zero_pad, spec = get_spectrogram_infer(audio_new, save_dir=None, length=length, save_results=False)
    else:
        audio_zero_pad, spec = get_spectrogram_non_normalize(audio_new, save_dir=None, length=length, save_results=False)

    # specvqgan expects inputs to be in [-1, 1] but spectrograms are in [0, 1]
    spec = 2 * spec - 1

    if return_wav:
        return audio_zero_pad, spec
    else:
        return spec


def show_grid(imgs):
    print('Rendering the Plot with Frames Used in Conditioning')
    figsize = ((imgs.shape[1] // 228 + 1) * 5, (imgs.shape[2] // 228 + 1) * 5)
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig

def calculate_codebook_bitrate(duration, quant_z, codebook_size):
    # Calculating the Bitrate
    bottle_neck_size = quant_z.shape[-2:]
    bits_per_codebook_entry = (codebook_size-1).bit_length()
    bitrate = bits_per_codebook_entry * bottle_neck_size.numel() / duration / 1024
    print(f'The input audio is {duration:.2f} seconds long.')
    print(f'Codebook size is {codebook_size} i.e. a codebook entry allocates {bits_per_codebook_entry} bits')
    print(f'SpecVQGAN bottleneck size: {list(bottle_neck_size)}')
    print(f'Thus, bitrate is {bitrate:.2f} kbps')
    return bitrate

def get_audio_file_bitrate(file):
    assert which_ffprobe() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    cmd = f'{which_ffprobe()} -v error -select_streams a:0'\
          f' -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 {file}'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    bitrate = int(result.stdout.decode('utf-8').replace('\n', ''))
    bitrate /= 1024
    return bitrate

def get_10s_video(video_path):
    video_name = video_path.split('/')[-1].split('_')[0]
    start_sec = float(video_path.split('/')[-1].split('_')[6])
    full_video_path = os.path.join('data/greatesthit/greatesthit_orig', video_name + '_denoised.mp4')
    trimmed_10s = trim_video(full_video_path, start_sec, 10)
    return trimmed_10s


if __name__ == '__main__':
    # if empty, it wasn't found
    print(which_ffmpeg())
