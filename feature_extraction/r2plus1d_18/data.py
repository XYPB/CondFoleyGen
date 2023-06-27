from utils import cropAudio, pick_async_frame_idx
from transforms import *
import json
from PIL import Image
import os
from tqdm import tqdm
import copy
import torch
from torchvision import transforms
import numpy as np
from numpy.random import randint
from random import sample
from torch.utils.data import Dataset
import torchaudio
torchaudio.set_audio_backend("sox_io")

SR = 16000
MAX_SAMPLE_ITER = 10


def get_transform():
    vis_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    vis_transform_vate = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    aud_transform = transforms.Compose([transforms.ToTensor()])
    return vis_transform, vis_transform_vate, aud_transform


def get_transform3D():
    vis_transform = transforms.Compose([
        Resize3D(128),
        RandomResizedCrop3D(112, scale=(0.5, 1.0)),
        RandomHorizontalFlip3D(),
        ColorJitter3D(brightness=0.1, saturation=0.1),
        ToTensor3D(),
        Normalize3D(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    vis_transform_vate = transforms.Compose([
        Resize3D(128),
        CenterCrop3D(112),
        ToTensor3D(),
        Normalize3D(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    aud_transform = transforms.Compose([transforms.ToTensor()])
    return vis_transform, vis_transform_vate, aud_transform


def non_negative(x): return int(np.round(max(0, x), 0))


class GreatestHit(Dataset):

    def __init__(self, root, video_list, frame_transform=None, wave_transform=None, fps=10, return_frame_path=False,
                 length=1., gap=0, n_negative=5, conditional=False, type_identify=False, iter_all_hit=False, drop_none=False,
                 return_type=False, **kwargs):
        super().__init__()
        self.root = root
        self.video_list = video_list
        self.frame_transform = frame_transform
        self.wave_transform = wave_transform
        self.fps = fps
        self.L = length
        self.gap = gap
        self.n_nega = n_negative
        self.cond = conditional
        self.return_frame_path = return_frame_path
        self.shift_range = set(
            list(range(int(-self.fps//2+1), int(self.fps//2-1)))) - set([0])
        self.type_identify = type_identify
        self.iter_all_hit = iter_all_hit
        self.return_type = return_type

        self.video_dict = {}
        self.hit_list = []
        all_type_cnt = {}
        for v in tqdm(self.video_list):
            record_file = os.path.join(self.root, v, 'hit_record.json')
            frame_cnt = len(os.listdir(os.path.join(root, v, 'frames')))
            hit_record = json.load(open(record_file, 'r'))
            # make sure None is consistent
            for i in range(len(hit_record)):
                if 'None' in hit_record[i][1]:
                    hit_record[i][1] = 'None None'
            if drop_none:
                hit_record = [item for item in hit_record if 'None' not in item[1]]
            if len(hit_record) == 0:
                continue
            max_length = np.max([t for t, _ in hit_record])
            # make sure each type of action contains more than 1 samples so the there are
            # samples for both target and conditional
            type_cnt = {}
            for _, hitType in hit_record:
                if hitType not in type_cnt.keys():
                    type_cnt[hitType] = 0
                if hitType not in all_type_cnt.keys():
                    all_type_cnt[hitType] = 0
                type_cnt[hitType] += 1
                all_type_cnt[hitType] += 1

            hit_record = [[t, hitType]
                          for t, hitType in hit_record if type_cnt[hitType] > 1]
            # remove videos with only one type of action:
            remain_type = [t for t, cnt in type_cnt.items() if cnt > 1]
            if len(remain_type) < 2:
                continue

            if frame_cnt < max_length * self.fps or len(hit_record) < self.n_nega + 2:
                continue
            self.video_dict[v] = [hit_record, frame_cnt]
            for i in range(len(hit_record)):
                self.hit_list.append([v, i, frame_cnt])
        print(f'Validated video size {len(self.video_dict)}')

        if self.iter_all_hit:
            self.size = len(self.hit_list)
            self.type_id = {k:i for i, k in enumerate(all_type_cnt.keys())}
        else:
            self.size = len(self.video_dict)
            self.video_list = [k for k in self.video_dict.keys()]
        print(all_type_cnt)
        print(self.size)

    def __len__(self):
        return self.size

    def _temporal_shift(self, t):
        shift = sample(self.shift_range, k=1)[0]
        return (np.round(self.fps * t) + shift) / self.fps

    def __getitem__(self, index):
        if self.iter_all_hit:
            v, tar_idx, n_frames = self.hit_list[index]
            hit_record = self.video_dict[v][0]
        else:
            v = self.video_list[index]
            hit_record, n_frames = self.video_dict[v]

        a = os.path.join(self.root, v, 'audio', f'{v}_denoised.wav')
        f = os.path.join(self.root, v, 'frames')
        # audio, sr = torchaudio.load(a)
        # if len(audio.shape) > 1:
        #     audio = torch.mean(audio, axis=0)
        # if sr != SR:
        #     resampler = torchaudio.transforms.Resample(sr, SR)
        #     audio = resampler(audio)
        #     sr = SR

        candidates = set(list(range(len(hit_record))))

        not_none_candidates = set(
            [i for i in range(len(hit_record)) if 'None' not in hit_record[i][1]])
        try:
            if not self.iter_all_hit:
                tar_idx = sample(not_none_candidates, k=1)[0]
                not_none_candidates.remove(tar_idx)
        except:
            print(hit_record, not_none_candidates)
            raise
        candidates.remove(tar_idx)
        tar_ct, tar_type = hit_record[tar_idx]

        selected_idx = []
        # pick conditional pair

        # test code
        # make sure there is always a conditional smaple with the same type
        remain_types = [hit_record[i][1] for i in candidates]
        assert tar_type in set(
            remain_types), 'Error: Target type occur for once only!'

        # if self.cond:
        #     # for _ in range(MAX_SAMPLE_ITER):
        #     while True:
        #         if self.iter_all_hit:
        #             cond_idx = sample(candidates, k=1)[0]
        #         else:
        #             cond_idx = sample(not_none_candidates, k=1)[0]
        #         cond_ct, cond_type = hit_record[cond_idx]
        #         if cond_type == tar_type:
        #             break
        #     candidates.remove(cond_idx)
        #     # shift conditional info
        #     # cond_ct = self._temporal_shift(cond_ct)

        #     cond_s = non_negative(
        #         self.fps * (cond_ct - self.L/2))
        #     cond_sp = non_negative(
        #         self.fps * (cond_ct - 1.5 * self.L))
        #     cond_e = non_negative(cond_s + self.fps)
        #     cond_frames = [Image.open(os.path.join(
        #         f, f'frame{i+1:0>6d}.jpg')).convert('RGB') for i in range(cond_s, cond_e)]
        #     cond_audio = cropAudio(
        #         audio, sr, cond_s, fps=self.fps, length=self.L, left_shift=0)
        #     selected_idx.append(cond_s)

        # pick target pair
        tar_s = non_negative(self.fps * (tar_ct - self.L/2))
        tar_sp = non_negative(self.fps * (tar_ct - 1.5 * self.L))
        tar_e = non_negative(tar_s + self.fps)
        tar_frames = [Image.open(os.path.join(
            f, f'frame{i+1:0>6d}.jpg')).convert('RGB') for i in range(tar_s, tar_e)]
        # tar_audio = cropAudio(
        #     audio, sr, tar_s, fps=self.fps, length=self.L, left_shift=0)
        # selected_idx.append(tar_s)

        # remove samples of the same type if type_identify
        for i in copy.copy(candidates):
            if hit_record[i][1] == tar_type:
                candidates.remove(i)

        # pick negative audio
        nega_audios = []
        for _ in range(self.n_nega):
            try:
                nega_idx = sample(candidates, k=1)[0]
            except:
                # repeat when there is no enough negatives
                nega_audios.append(nega_audios[-1])
                # print(hit_record)
                # print(candidates, tar_type)
                # print(v)
                continue
            nega_ct, nega_type = hit_record[nega_idx]
            if not self.type_identify and nega_type == tar_type:
                nega_ct = self._temporal_shift(nega_ct)
            nega_s = non_negative(
                self.fps * (nega_ct - self.L/2))
            # nega_audios.append(
            #     cropAudio(audio, sr, nega_s, fps=self.fps, length=self.L, left_shift=0))

            candidates.remove(nega_idx)
            selected_idx.append(nega_s)
        # except Exception as e:
        #   print(f'==> {e}: {v} failed!')
        #   exit(1)

        if self.frame_transform != None:
            # if self.cond:
                # cond_frames = self.frame_transform(cond_frames)
                # cond_frames = torch.stack(cond_frames, dim=1)
            tar_frames = self.frame_transform(tar_frames)
            tar_frames = torch.stack(tar_frames, dim=1)

        # if self.wave_transform != None:
        #     if self.cond:
        #         cond_audio = self.wave_transform(cond_audio)
        #     tar_audio = self.wave_transform(tar_audio)
        #     nega_audios = [self.wave_transform(a) for a in nega_audios]
        # train_audio = [tar_audio] + nega_audios


        cond_frames = None
        cond_audio = None
        train_audio = None
        cond_ct = None
        cond_type = None
        cond_idx = None
        return tar_frames, [v, tar_ct, tar_type]
        if self.cond and not self.return_frame_path and not self.iter_all_hit:
            return cond_frames, cond_audio, tar_frames, train_audio
        elif self.return_frame_path:
            return cond_frames, cond_audio, tar_frames, train_audio, v, selected_idx
        elif self.cond and self.iter_all_hit:
            assert tar_idx != cond_idx
            # if cond_type != tar_type:
            #     print(cond_type, tar_type, hit_record)
            if self.return_type:
                return cond_frames, cond_audio, tar_frames, train_audio, self.type_id[tar_type]
            else:
                return cond_frames, cond_audio, tar_frames, train_audio, [v, str(tar_ct), tar_type, str(cond_ct), cond_type]
        else:
            return tar_frames, train_audio

