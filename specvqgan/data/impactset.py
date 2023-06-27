import json
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from random import sample
import torchaudio
import logging
from glob import glob
import sys
import soundfile
import copy
import csv
import noisereduce as nr

sys.path.insert(0, '.')  # nopep8
from train import instantiate_from_config
from specvqgan.data.transforms import *

torchaudio.set_audio_backend("sox_io")
logger = logging.getLogger(f'main.{__name__}')

SR = 22050
FPS = 15
MAX_SAMPLE_ITER = 10

def non_negative(x): return int(np.round(max(0, x), 0))

def rms(x): return np.sqrt(np.mean(x**2))

def get_GH_data_identifier(video_name, start_idx, split='_'):
    if isinstance(start_idx, str):
        return video_name + split + start_idx
    elif isinstance(start_idx, int):
        return video_name + split + str(start_idx)
    else:
        raise NotImplementedError

def draw_spec(spec, dest, cmap='magma'):
    plt.imshow(spec, cmap=cmap, origin='lower')
    plt.axis('off')
    plt.savefig(dest, bbox_inches='tight', pad_inches=0., dpi=300)
    plt.close()

def convert_to_decibel(arr):
    ref = 1
    return 20 * np.log10(abs(arr + 1e-4) / ref)

class ResampleFrames(object):
    def __init__(self, feat_sample_size, times_to_repeat_after_resample=None):
        self.feat_sample_size = feat_sample_size
        self.times_to_repeat_after_resample = times_to_repeat_after_resample

    def __call__(self, item):
        feat_len = item['feature'].shape[0]

        ## resample
        assert feat_len >= self.feat_sample_size
        # evenly spaced points (abcdefghkl -> aoooofoooo)
        idx = np.linspace(0, feat_len, self.feat_sample_size, dtype=np.int, endpoint=False)
        # xoooo xoooo -> ooxoo ooxoo
        shift = feat_len // (self.feat_sample_size + 1)
        idx = idx + shift

        ## repeat after resampling (abc -> aaaabbbbcccc)
        if self.times_to_repeat_after_resample is not None and self.times_to_repeat_after_resample > 1:
            idx = np.repeat(idx, self.times_to_repeat_after_resample)

        item['feature'] = item['feature'][idx, :]
        return item


class ImpactSetWave(torch.utils.data.Dataset):

    def __init__(self, split, random_crop, mel_num, spec_crop_len,
                L=2.0, denoise=False, splits_path='./data',
                data_path='data/ImpactSet/impactset-proccess-resize'):
        super().__init__()
        self.split = split
        self.splits_path = splits_path
        self.data_path = data_path
        self.L = L
        self.denoise = denoise

        video_name_split_path = os.path.join(splits_path, f'countixAV_{split}.json')
        if not os.path.exists(video_name_split_path):
            self.make_split_files()
        video_name = json.load(open(video_name_split_path, 'r'))
        self.video_frame_cnt = {v: len(os.listdir(os.path.join(self.data_path, v, 'frames'))) for v in video_name}
        self.left_over = int(FPS * L + 1)
        self.video_audio_path = {v: os.path.join(self.data_path, v, f'audio/{v}_resampled.wav') for v in video_name}
        self.dataset = video_name

        self.wav_transforms = transforms.Compose([
            MakeMono(),
            Padding(target_len=int(SR * self.L)),
        ])
        
        self.spec_transforms = CropImage([mel_num, spec_crop_len], random_crop)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {}
        video = self.dataset[idx]

        available_frame_idx = self.video_frame_cnt[video] - self.left_over
        wav = None
        spec = None
        max_db = -np.inf
        wave_path = ''
        cur_wave_path = self.video_audio_path[video]
        if self.denoise:
            cur_wave_path = cur_wave_path.replace('.wav', '_denoised.wav')
        for _ in range(10):
            start_idx = torch.randint(0, available_frame_idx, (1,)).tolist()[0]
            # target
            start_t = (start_idx + 0.5) / FPS
            start_audio_idx = non_negative(start_t * SR)

            cur_wav, _ = soundfile.read(cur_wave_path, frames=int(SR * self.L), start=start_audio_idx)

            decibel = convert_to_decibel(cur_wav)
            if float(np.mean(decibel)) > max_db:
                wav = cur_wav
                wave_path = cur_wave_path
                max_db = float(np.mean(decibel))
            if max_db >= -40:
                break

        # print(max_db)
        wav = self.wav_transforms(wav)
        item['image'] = wav # (80, 173)
        # item['wav'] = wav
        item['file_path_wav_'] = wave_path

        item['label'] = 'None'
        item['target'] = 'None'

        return item

    def make_split_files(self):
        raise NotImplementedError

class ImpactSetWaveTrain(ImpactSetWave):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)

class ImpactSetWaveValidation(ImpactSetWave):
    def __init__(self, specs_dataset_cfg):
        super().__init__('val', **specs_dataset_cfg)

class ImpactSetWaveTest(ImpactSetWave):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)


class ImpactSetSpec(torch.utils.data.Dataset):

    def __init__(self, split, random_crop, mel_num, spec_crop_len,
                L=2.0, denoise=False, splits_path='./data',
                data_path='data/ImpactSet/impactset-proccess-resize'):
        super().__init__()
        self.split = split
        self.splits_path = splits_path
        self.data_path = data_path
        self.L = L
        self.denoise = denoise

        video_name_split_path = os.path.join(splits_path, f'countixAV_{split}.json')
        if not os.path.exists(video_name_split_path):
            self.make_split_files()
        video_name = json.load(open(video_name_split_path, 'r'))
        self.video_frame_cnt = {v: len(os.listdir(os.path.join(self.data_path, v, 'frames'))) for v in video_name}
        self.left_over = int(FPS * L + 1)
        self.video_audio_path = {v: os.path.join(self.data_path, v, f'audio/{v}_resampled.wav') for v in video_name}
        self.dataset = video_name

        self.wav_transforms = transforms.Compose([
            MakeMono(),
            SpectrogramTorchAudio(nfft=1024, hoplen=1024//4, spec_power=1),
            MelScaleTorchAudio(sr=SR, stft=513, fmin=125, fmax=7600, nmels=80),
            LowerThresh(1e-5),
            Log10(),
            Multiply(20),
            Subtract(20),
            Add(100),
            Divide(100),
            Clip(0, 1.0),
            TrimSpec(173),
        ])
        
        self.spec_transforms = CropImage([mel_num, spec_crop_len], random_crop)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {}
        video = self.dataset[idx]

        available_frame_idx = self.video_frame_cnt[video] - self.left_over
        wav = None
        spec = None
        max_rms = -np.inf
        wave_path = ''
        cur_wave_path = self.video_audio_path[video]
        if self.denoise:
            cur_wave_path = cur_wave_path.replace('.wav', '_denoised.wav')
        for _ in range(10):
            start_idx = torch.randint(0, available_frame_idx, (1,)).tolist()[0]
            # target
            start_t = (start_idx + 0.5) / FPS
            start_audio_idx = non_negative(start_t * SR)

            cur_wav, _ = soundfile.read(cur_wave_path, frames=int(SR * self.L), start=start_audio_idx)

            if self.wav_transforms is not None:
                spec_tensor = self.wav_transforms(torch.tensor(cur_wav).float())
                cur_spec = spec_tensor.numpy()
            # zeros padding if not enough spec t steps
            if cur_spec.shape[1] < 173:
                pad = np.zeros((80, 173), dtype=cur_spec.dtype)
                pad[:, :cur_spec.shape[1]] = cur_spec
                cur_spec = pad
            rms_val = rms(cur_spec)
            if rms_val > max_rms:
                wav = cur_wav
                spec = cur_spec
                wave_path = cur_wave_path
                max_rms = rms_val
            # print(rms_val)
            if max_rms >= 0.1:
                break

        item['image'] = 2 * spec - 1 # (80, 173)
        # item['wav'] = wav
        item['file_path_wav_'] = wave_path

        item['label'] = 'None'
        item['target'] = 'None'

        if self.spec_transforms is not None:
            item = self.spec_transforms(item)
        return item

    def make_split_files(self):
        raise NotImplementedError

class ImpactSetSpecTrain(ImpactSetSpec):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)

class ImpactSetSpecValidation(ImpactSetSpec):
    def __init__(self, specs_dataset_cfg):
        super().__init__('val', **specs_dataset_cfg)

class ImpactSetSpecTest(ImpactSetSpec):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)



class ImpactSetWaveTestTime(torch.utils.data.Dataset):

    def __init__(self, split, random_crop, mel_num, spec_crop_len,
                L=2.0, denoise=False, splits_path='./data',
                data_path='data/ImpactSet/impactset-proccess-resize'):
        super().__init__()
        self.split = split
        self.splits_path = splits_path
        self.data_path = data_path
        self.L = L
        self.denoise = denoise

        self.video_list = glob('data/ImpactSet/RawVideos/StockVideo_sound/*.wav') + [
            'data/ImpactSet/RawVideos/YouTube-impact-ccl/1_ckbCU5aQs/1_ckbCU5aQs_0013_0016_resize.wav', 
            'data/ImpactSet/RawVideos/YouTube-impact-ccl/GFmuVBiwz6k/GFmuVBiwz6k_0034_0054_resize.wav',
            'data/ImpactSet/RawVideos/YouTube-impact-ccl/OsPcY316h1M/OsPcY316h1M_0000_0005_resize.wav',
            'data/ImpactSet/RawVideos/YouTube-impact-ccl/SExIpBIBj_k/SExIpBIBj_k_0009_0019_resize.wav',
            'data/ImpactSet/RawVideos/YouTube-impact-ccl/S6TkbV4B4QI/S6TkbV4B4QI_0028_0036_resize.wav',
            'data/ImpactSet/RawVideos/YouTube-impact-ccl/2Ld24pPIn3k/2Ld24pPIn3k_0005_0011_resize.wav',
            'data/ImpactSet/RawVideos/YouTube-impact-ccl/6d1YS7fdBK4/6d1YS7fdBK4_0007_0019_resize.wav',
            'data/ImpactSet/RawVideos/YouTube-impact-ccl/JnBsmJgEkiw/JnBsmJgEkiw_0008_0016_resize.wav',
            'data/ImpactSet/RawVideos/YouTube-impact-ccl/xcUyiXt0gjo/xcUyiXt0gjo_0015_0021_resize.wav',
            'data/ImpactSet/RawVideos/YouTube-impact-ccl/4DRFJnZjpMM/4DRFJnZjpMM_0000_0010_resize.wav'
        ] + glob('data/ImpactSet/RawVideos/self_recorded/*_resize.wav')

        self.wav_transforms = transforms.Compose([
            MakeMono(),
            SpectrogramTorchAudio(nfft=1024, hoplen=1024//4, spec_power=1),
            MelScaleTorchAudio(sr=SR, stft=513, fmin=125, fmax=7600, nmels=80),
            LowerThresh(1e-5),
            Log10(),
            Multiply(20),
            Subtract(20),
            Add(100),
            Divide(100),
            Clip(0, 1.0),
            TrimSpec(173),
        ])
        self.spec_transforms = CropImage([mel_num, spec_crop_len], random_crop)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        item = {}

        wave_path = self.video_list[idx]

        wav, _ = soundfile.read(wave_path)
        start_idx = random.randint(0, min(4, wav.shape[0] - int(SR * self.L)))
        wav = wav[start_idx:start_idx+int(SR * self.L)]

        if self.denoise:
            if len(wav.shape) == 1:
                wav = wav[None, :]
            wav = nr.reduce_noise(y=wav, sr=SR, n_fft=1024, hop_length=1024//4)
            wav = wav.squeeze()
        if self.wav_transforms is not None:
            spec_tensor = self.wav_transforms(torch.tensor(wav).float())
            spec = spec_tensor.numpy()
        if spec.shape[1] < 173:
            pad = np.zeros((80, 173), dtype=spec.dtype)
            pad[:, :spec.shape[1]] = spec
            spec = pad

        item['image'] = 2 * spec - 1 # (80, 173)
        # item['wav'] = wav
        item['file_path_wav_'] = wave_path

        item['label'] = 'None'
        item['target'] = 'None'

        if self.spec_transforms is not None:
            item = self.spec_transforms(item)
        return item

    def make_split_files(self):
        raise NotImplementedError

class ImpactSetWaveTestTimeTrain(ImpactSetWaveTestTime):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)

class ImpactSetWaveTestTimeValidation(ImpactSetWaveTestTime):
    def __init__(self, specs_dataset_cfg):
        super().__init__('val', **specs_dataset_cfg)

class ImpactSetWaveTestTimeTest(ImpactSetWaveTestTime):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)


class ImpactSetWaveWithSilent(torch.utils.data.Dataset):

    def __init__(self, split, random_crop, mel_num, spec_crop_len,
                L=2.0, denoise=False, splits_path='./data',
                data_path='data/ImpactSet/impactset-proccess-resize'):
        super().__init__()
        self.split = split
        self.splits_path = splits_path
        self.data_path = data_path
        self.L = L
        self.denoise = denoise

        video_name_split_path = os.path.join(splits_path, f'countixAV_{split}.json')
        if not os.path.exists(video_name_split_path):
            self.make_split_files()
        video_name = json.load(open(video_name_split_path, 'r'))
        self.video_frame_cnt = {v: len(os.listdir(os.path.join(self.data_path, v, 'frames'))) for v in video_name}
        self.left_over = int(FPS * L + 1)
        self.video_audio_path = {v: os.path.join(self.data_path, v, f'audio/{v}_resampled.wav') for v in video_name}
        self.dataset = video_name

        self.wav_transforms = transforms.Compose([
            MakeMono(),
            Padding(target_len=int(SR * self.L)),
        ])
        
        self.spec_transforms = CropImage([mel_num, spec_crop_len], random_crop)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {}
        video = self.dataset[idx]

        available_frame_idx = self.video_frame_cnt[video] - self.left_over
        wave_path = self.video_audio_path[video]
        if self.denoise:
            wave_path = wave_path.replace('.wav', '_denoised.wav')
        start_idx = torch.randint(0, available_frame_idx, (1,)).tolist()[0]
        # target
        start_t = (start_idx + 0.5) / FPS
        start_audio_idx = non_negative(start_t * SR)

        wav, _ = soundfile.read(wave_path, frames=int(SR * self.L), start=start_audio_idx)

        wav = self.wav_transforms(wav)

        item['image'] = wav # (44100,)
        # item['wav'] = wav
        item['file_path_wav_'] = wave_path

        item['label'] = 'None'
        item['target'] = 'None'
        return item

    def make_split_files(self):
        raise NotImplementedError

class ImpactSetWaveWithSilentTrain(ImpactSetWaveWithSilent):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)

class ImpactSetWaveWithSilentValidation(ImpactSetWaveWithSilent):
    def __init__(self, specs_dataset_cfg):
        super().__init__('val', **specs_dataset_cfg)

class ImpactSetWaveWithSilentTest(ImpactSetWaveWithSilent):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)


class ImpactSetWaveCondOnImage(torch.utils.data.Dataset):

    def __init__(self, split,
                L=2.0, frame_transforms=None, denoise=False, splits_path='./data',
                data_path='data/ImpactSet/impactset-proccess-resize',
                p_outside_cond=0.):
        super().__init__()
        self.split = split
        self.splits_path = splits_path
        self.frame_transforms = frame_transforms
        self.data_path = data_path
        self.L = L
        self.denoise = denoise
        self.p_outside_cond = torch.tensor(p_outside_cond)

        video_name_split_path = os.path.join(splits_path, f'countixAV_{split}.json')
        if not os.path.exists(video_name_split_path):
            self.make_split_files()
        video_name = json.load(open(video_name_split_path, 'r'))
        self.video_frame_cnt = {v: len(os.listdir(os.path.join(self.data_path, v, 'frames'))) for v in video_name}
        self.left_over = int(FPS * L + 1)
        for v, cnt in self.video_frame_cnt.items():
            if cnt - (3*self.left_over) <= 0:
                video_name.remove(v)
        self.video_audio_path = {v: os.path.join(self.data_path, v, f'audio/{v}_resampled.wav') for v in video_name}
        self.dataset = video_name

        video_timing_split_path = os.path.join(splits_path, f'countixAV_{split}_timing.json')
        self.video_timing = json.load(open(video_timing_split_path, 'r'))
        self.video_timing = {v: [int(float(t) * FPS) for t in ts] for v, ts in self.video_timing.items()}

        if split != 'test':
            video_class_path = os.path.join(splits_path, f'countixAV_{split}_class.json')
            if not os.path.exists(video_class_path):
                self.make_video_class()
            self.video_class = json.load(open(video_class_path, 'r'))
            self.class2video = {}
            for v, c in self.video_class.items():
                if c not in self.class2video.keys():
                    self.class2video[c] = []
                self.class2video[c].append(v)

        self.wav_transforms = transforms.Compose([
            MakeMono(),
            Padding(target_len=int(SR * self.L)),
        ])
        if self.frame_transforms == None:
            self.frame_transforms = transforms.Compose([
                Resize3D(128),
                RandomResizedCrop3D(112, scale=(0.5, 1.0)),
                RandomHorizontalFlip3D(),
                ColorJitter3D(brightness=0.1, saturation=0.1),
                ToTensor3D(),
                Normalize3D(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    def make_video_class(self):
        meta_path = f'data/ImpactSet/data-info/CountixAV_{self.split}.csv'
        video_class = {}
        with open(meta_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                vid, k_st, k_et = row[:3]
                video_name = f'{vid}_{int(k_st):0>4d}_{int(k_et):0>4d}'
                if video_name not in self.dataset:
                    continue
                video_class[video_name] = row[-1]
        with open(os.path.join(self.splits_path, f'countixAV_{self.split}_class.json'), 'w') as f:
            json.dump(video_class, f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {}
        video = self.dataset[idx]

        available_frame_idx = self.video_frame_cnt[video] - self.left_over
        rep_start_idx, rep_end_idx = self.video_timing[video]
        rep_end_idx = min(available_frame_idx, rep_end_idx)
        if available_frame_idx <= rep_start_idx + self.L * FPS:
            idx_set = list(range(0, available_frame_idx))
        else:
            idx_set = list(range(rep_start_idx, rep_end_idx))
        start_idx = sample(idx_set, k=1)[0]

        wave_path = self.video_audio_path[video]
        if self.denoise:
            wave_path = wave_path.replace('.wav', '_denoised.wav')

        # target
        start_t = (start_idx + 0.5) / FPS
        end_idx= non_negative(start_idx + FPS * self.L)
        start_audio_idx = non_negative(start_t * SR)
        wav, sr = soundfile.read(wave_path, frames=int(SR * self.L), start=start_audio_idx)
        assert sr == SR
        wav = self.wav_transforms(wav)
        frame_path = os.path.join(self.data_path, video, 'frames')
        frames = [Image.open(os.path.join(
            frame_path, f'frame{i+1:0>6d}.jpg')).convert('RGB') for i in
            range(start_idx, end_idx)]

        if torch.all(torch.bernoulli(self.p_outside_cond) == 1.) and self.split != 'test':
            # outside from the same class
            cur_class = self.video_class[video]
            tmp_video = copy.copy(self.class2video[cur_class])
            if len(tmp_video) > 1:
                # if only 1 video in the class, use itself
                tmp_video.remove(video)
            cond_video = sample(tmp_video, k=1)[0]
            cond_available_frame_idx = self.video_frame_cnt[cond_video] - self.left_over
            cond_start_idx = torch.randint(0, cond_available_frame_idx, (1,)).tolist()[0]
        else:
            cond_video = video
            idx_set = list(range(0, start_idx)) + list(range(end_idx, available_frame_idx))
            cond_start_idx = random.sample(idx_set, k=1)[0]

        cond_end_idx = non_negative(cond_start_idx + FPS * self.L)
        cond_start_t = (cond_start_idx + 0.5) / FPS
        cond_audio_idx = non_negative(cond_start_t * SR)
        cond_frame_path = os.path.join(self.data_path, cond_video, 'frames')
        cond_wave_path = self.video_audio_path[cond_video]

        cond_frames = [Image.open(os.path.join(
            cond_frame_path, f'frame{i+1:0>6d}.jpg')).convert('RGB') for i in
            range(cond_start_idx, cond_end_idx)]
        cond_wav, sr = soundfile.read(cond_wave_path, frames=int(SR * self.L), start=cond_audio_idx)
        assert sr == SR
        cond_wav = self.wav_transforms(cond_wav)

        item['image'] = wav # (44100,)
        item['cond_image'] = cond_wav # (44100,)
        item['file_path_wav_'] = wave_path
        item['file_path_cond_wav_'] = cond_wave_path

        if self.frame_transforms is not None:
            cond_frames = self.frame_transforms(cond_frames)
            frames = self.frame_transforms(frames)

        item['feature'] = np.stack(cond_frames + frames, axis=0) # (30 * L, 112, 112, 3)
        item['file_path_feats_'] = (frame_path, start_idx)
        item['file_path_cond_feats_'] = (cond_frame_path, cond_start_idx)

        item['label'] = 'None'
        item['target'] = 'None'

        return item

    def make_split_files(self):
        raise NotImplementedError


class ImpactSetWaveCondOnImageTrain(ImpactSetWaveCondOnImage):
    def __init__(self, dataset_cfg):
        train_transforms = transforms.Compose([
            Resize3D(128),
            RandomResizedCrop3D(112, scale=(0.5, 1.0)),
            RandomHorizontalFlip3D(),
            ColorJitter3D(brightness=0.4, saturation=0.4, contrast=0.2, hue=0.1),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        super().__init__('train', frame_transforms=train_transforms, **dataset_cfg)

class ImpactSetWaveCondOnImageValidation(ImpactSetWaveCondOnImage):
    def __init__(self, dataset_cfg):
        valid_transforms = transforms.Compose([
            Resize3D(128),
            CenterCrop3D(112),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        super().__init__('val', frame_transforms=valid_transforms, **dataset_cfg)

class ImpactSetWaveCondOnImageTest(ImpactSetWaveCondOnImage):
    def __init__(self, dataset_cfg):
        test_transforms = transforms.Compose([
            Resize3D(128),
            CenterCrop3D(112),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        super().__init__('test', frame_transforms=test_transforms, **dataset_cfg)



class ImpactSetCleanWaveCondOnImage(ImpactSetWaveCondOnImage):
    def __init__(self, split, L=2, frame_transforms=None, denoise=False, splits_path='./data', data_path='data/ImpactSet/impactset-proccess-resize', p_outside_cond=0):
        super().__init__(split, L, frame_transforms, denoise, splits_path, data_path, p_outside_cond)
        pred_timing_path = f'data/countixAV_{split}_timing_processed_0.20.json'
        assert os.path.exists(pred_timing_path)
        self.pred_timing = json.load(open(pred_timing_path, 'r'))

        self.dataset = []
        for v, ts in self.pred_timing.items():
            if v in self.video_audio_path.keys():
                for t in ts:
                    self.dataset.append([v, t])

    def __getitem__(self, idx):
        item = {}
        video, start_t = self.dataset[idx]
        available_frame_idx = self.video_frame_cnt[video] - self.left_over
        available_timing = (available_frame_idx + 0.5) / FPS
        start_t = float(start_t)
        start_t = min(start_t, available_timing)

        start_idx = non_negative(start_t * FPS - 0.5)

        wave_path = self.video_audio_path[video]
        if self.denoise:
            wave_path = wave_path.replace('.wav', '_denoised.wav')

        # target
        end_idx= non_negative(start_idx + FPS * self.L)
        start_audio_idx = non_negative(start_t * SR)
        wav, sr = soundfile.read(wave_path, frames=int(SR * self.L), start=start_audio_idx)
        assert sr == SR
        wav = self.wav_transforms(wav)
        frame_path = os.path.join(self.data_path, video, 'frames')
        frames = [Image.open(os.path.join(
            frame_path, f'frame{i+1:0>6d}.jpg')).convert('RGB') for i in
            range(start_idx, end_idx)]

        if torch.all(torch.bernoulli(self.p_outside_cond) == 1.):
            other_video = list(self.pred_timing.keys())
            other_video.remove(video)
            cond_video = sample(other_video, k=1)[0]
            cond_available_frame_idx = self.video_frame_cnt[cond_video] - self.left_over
            cond_available_timing = (cond_available_frame_idx + 0.5) / FPS
        else:
            cond_video = video
            cond_available_timing = available_timing

        cond_start_t = sample(self.pred_timing[cond_video], k=1)[0]
        cond_start_t = float(cond_start_t)
        cond_start_t = min(cond_start_t, cond_available_timing)
        cond_start_idx = non_negative(cond_start_t * FPS - 0.5)
        cond_end_idx = non_negative(cond_start_idx + FPS * self.L)
        cond_audio_idx = non_negative(cond_start_t * SR)
        cond_frame_path = os.path.join(self.data_path, cond_video, 'frames')
        cond_wave_path = self.video_audio_path[cond_video]

        cond_frames = [Image.open(os.path.join(
            cond_frame_path, f'frame{i+1:0>6d}.jpg')).convert('RGB') for i in
            range(cond_start_idx, cond_end_idx)]
        cond_wav, sr = soundfile.read(cond_wave_path, frames=int(SR * self.L), start=cond_audio_idx)
        assert sr == SR
        cond_wav = self.wav_transforms(cond_wav)

        item['image'] = wav # (44100,)
        item['cond_image'] = cond_wav # (44100,)
        item['file_path_wav_'] = wave_path
        item['file_path_cond_wav_'] = cond_wave_path

        if self.frame_transforms is not None:
            cond_frames = self.frame_transforms(cond_frames)
            frames = self.frame_transforms(frames)

        item['feature'] = np.stack(cond_frames + frames, axis=0) # (30 * L, 112, 112, 3)
        item['file_path_feats_'] = (frame_path, start_idx)
        item['file_path_cond_feats_'] = (cond_frame_path, cond_start_idx)

        item['label'] = 'None'
        item['target'] = 'None'

        return item


class ImpactSetCleanWaveCondOnImageTrain(ImpactSetCleanWaveCondOnImage):
    def __init__(self, dataset_cfg):
        train_transforms = transforms.Compose([
            Resize3D(128),
            RandomResizedCrop3D(112, scale=(0.5, 1.0)),
            RandomHorizontalFlip3D(),
            ColorJitter3D(brightness=0.4, saturation=0.4, contrast=0.2, hue=0.1),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        super().__init__('train', frame_transforms=train_transforms, **dataset_cfg)

class ImpactSetCleanWaveCondOnImageValidation(ImpactSetCleanWaveCondOnImage):
    def __init__(self, dataset_cfg):
        valid_transforms = transforms.Compose([
            Resize3D(128),
            CenterCrop3D(112),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        super().__init__('val', frame_transforms=valid_transforms, **dataset_cfg)

class ImpactSetCleanWaveCondOnImageTest(ImpactSetCleanWaveCondOnImage):
    def __init__(self, dataset_cfg):
        test_transforms = transforms.Compose([
            Resize3D(128),
            CenterCrop3D(112),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        super().__init__('test', frame_transforms=test_transforms, **dataset_cfg)


if __name__ == '__main__':
    import sys

    from omegaconf import OmegaConf
    cfg = OmegaConf.load('configs/countixAV_transformer_denoise_clean.yaml')
    data = instantiate_from_config(cfg.data)
    data.prepare_data()
    data.setup()

    print(data.datasets['train'])
    print(len(data.datasets['train']))
    # print(data.datasets['train'][24])
    exit()

    stats = []
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed = 0
    for k in range(1):
        x = np.arange(SR * 2)
        for i in tqdm(range(len(data.datasets['train']))):
            wav = data.datasets['train'][i]['wav']
            spec = data.datasets['train'][i]['image']
            spec = 0.5 * (spec + 1)
            spec_rms = rms(spec)
            stats.append(float(spec_rms))
            # plt.plot(x, wav)
            # plt.ylim(-1, 1)
            # plt.savefig(f'tmp/th0.1_wav_e_{k}_{i}_{mean_val:.3f}_{spec_rms:.3f}.png')
            # plt.close()
            # plt.cla()
            soundfile.write(f'tmp/wav_e_{k}_{i}_{spec_rms:.3f}.wav', wav, SR)
            draw_spec(spec, f'tmp/wav_spec_e_{k}_{i}_{spec_rms:.3f}.png')
            if i == 100:
                break
    # plt.hist(stats, bins=50)
    # plt.savefig(f'tmp/rms_spec_stats.png')
