from matplotlib import collections
import json
import os
import copy
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from random import sample
import torchaudio
import logging
import collections
from glob import glob
import sys
import albumentations
import soundfile

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


class Crop(object):

    def __init__(self, cropped_shape=None, random_crop=False):
        self.cropped_shape = cropped_shape
        if cropped_shape is not None:
            mel_num, spec_len = cropped_shape
            if random_crop:
                self.cropper = albumentations.RandomCrop
            else:
                self.cropper = albumentations.CenterCrop
            self.preprocessor = albumentations.Compose([self.cropper(mel_num, spec_len)])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __call__(self, item):
        item['image'] = self.preprocessor(image=item['image'])['image']
        if 'cond_image' in item.keys():
            item['cond_image'] = self.preprocessor(image=item['cond_image'])['image']
        return item

class CropImage(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

class CropFeats(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

    def __call__(self, item):
        item['feature'] = self.preprocessor(image=item['feature'])['image']
        return item

class CropCoords(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

    def __call__(self, item):
        item['coord'] = self.preprocessor(image=item['coord'])['image']
        return item

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


class GreatestHitSpecs(torch.utils.data.Dataset):

    def __init__(self, split, spec_dir_path, spec_len, random_crop, mel_num, 
                spec_crop_len, L=2.0, rand_shift=False, spec_transforms=None, splits_path='./data', 
                meta_path='./data/info_r2plus1d_dim1024_15fps.json'):
        super().__init__()
        self.split = split
        self.specs_dir = spec_dir_path
        self.spec_transforms = spec_transforms
        self.splits_path = splits_path
        self.meta_path = meta_path
        self.spec_len = spec_len
        self.rand_shift = rand_shift
        self.L = L
        self.spec_take_first = int(math.ceil(860 * (L / 10.) / 32) * 32)
        self.spec_take_first = 860 if self.spec_take_first > 860 else self.spec_take_first

        greatesthit_meta = json.load(open(self.meta_path, 'r'))
        unique_classes = sorted(list(set(ht for ht in greatesthit_meta['hit_type'])))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}
        self.target2label = {target: label for label, target in self.label2target.items()}
        self.video_idx2label = {
            get_GH_data_identifier(greatesthit_meta['video_name'][i], greatesthit_meta['start_idx'][i]): 
            greatesthit_meta['hit_type'][i] for i in range(len(greatesthit_meta['video_name']))
        }
        self.available_video_hit = list(self.video_idx2label.keys())
        self.video_idx2path = {
            vh: os.path.join(self.specs_dir, 
                vh.replace('_', '_denoised_') + '_' + self.video_idx2label[vh].replace(' ', '_') +'_mel.npy')
            for vh in self.available_video_hit
        }
        self.video_idx2idx = {
            get_GH_data_identifier(greatesthit_meta['video_name'][i], greatesthit_meta['start_idx'][i]):
            i for i in range(len(greatesthit_meta['video_name']))
        }

        split_clip_ids_path = os.path.join(splits_path, f'greatesthit_{split}.json')
        if not os.path.exists(split_clip_ids_path):
            raise NotImplementedError()
        clip_video_hit = json.load(open(split_clip_ids_path, 'r'))
        self.dataset = clip_video_hit
        spec_crop_len = self.spec_take_first if self.spec_take_first <= spec_crop_len else spec_crop_len
        self.spec_transforms = transforms.Compose([
            CropImage([mel_num, spec_crop_len], random_crop),
            # transforms.RandomApply([FrequencyMasking(freq_mask_param=20)], p=0),
            # transforms.RandomApply([TimeMasking(time_mask_param=int(32 * self.L))], p=0)
        ])

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {}

        video_idx = self.dataset[idx]
        spec_path = self.video_idx2path[video_idx]
        spec = np.load(spec_path) # (80, 860)

        if self.rand_shift:
            shift = random.uniform(0, 0.5)
            spec_shift = int(shift * spec.shape[1] // 10)
            # Since only the first second is used
            spec = np.roll(spec, -spec_shift, 1)

        # concat spec outside dataload
        item['image'] = 2 * spec - 1 # (80, 860)
        item['image'] = item['image'][:, :self.spec_take_first]
        item['file_path'] = spec_path

        item['label'] = self.video_idx2label[video_idx]
        item['target'] = self.label2target[item['label']]

        if self.spec_transforms is not None:
            item = self.spec_transforms(item)

        return item


class GreatestHitSpecsTrain(GreatestHitSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)

class GreatestHitSpecsValidation(GreatestHitSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('val', **specs_dataset_cfg)

class GreatestHitSpecsTest(GreatestHitSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)



class GreatestHitWave(torch.utils.data.Dataset):

    def __init__(self, split, wav_dir, random_crop, mel_num, spec_crop_len, spec_len,
                L=2.0, splits_path='./data', rand_shift=True,
                data_path='data/greatesthit/greatesthit-process-resized'):
        super().__init__()
        self.split = split
        self.wav_dir = wav_dir
        self.splits_path = splits_path
        self.data_path = data_path
        self.L = L
        self.rand_shift = rand_shift

        split_clip_ids_path = os.path.join(splits_path, f'greatesthit_{split}.json')
        if not os.path.exists(split_clip_ids_path):
            raise NotImplementedError()
        clip_video_hit = json.load(open(split_clip_ids_path, 'r'))

        video_name = list(set([vidx.split('_')[0] for vidx in clip_video_hit]))

        self.video_frame_cnt = {v: len(os.listdir(os.path.join(self.data_path, v, 'frames'))) // 2 for v in video_name}
        self.left_over = int(FPS * L + 1)
        self.video_audio_path = {v: os.path.join(self.data_path, v, f'audio/{v}_denoised_resampled.wav') for v in video_name}
        self.dataset = clip_video_hit

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

        self.wav_transforms = transforms.Compose([
            MakeMono(),
            Padding(target_len=int(SR * self.L)),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {}
        video_idx = self.dataset[idx]
        video, start_idx = video_idx.split('_')
        start_idx = int(start_idx)
        if self.rand_shift:
            shift = int(random.uniform(-0.5, 0.5) * SR)
            start_idx = non_negative(start_idx + shift)

        wave_path = self.video_audio_path[video]
        wav, sr = soundfile.read(wave_path, frames=int(SR * self.L), start=start_idx)
        assert sr == SR
        wav = self.wav_transforms(wav)

        item['image'] = wav # (44100,)
        # item['wav'] = wav
        item['file_path_wav_'] = wave_path

        item['label'] = 'None'
        item['target'] = 'None'

        return item


class GreatestHitWaveTrain(GreatestHitWave):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)

class GreatestHitWaveValidation(GreatestHitWave):
    def __init__(self, specs_dataset_cfg):
        super().__init__('val', **specs_dataset_cfg)

class GreatestHitWaveTest(GreatestHitWave):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)


class CondGreatestHitSpecsCondOnImage(torch.utils.data.Dataset):

    def __init__(self, split, specs_dir, spec_len, feat_len, feat_depth, feat_crop_len, random_crop, mel_num, spec_crop_len,
                vqgan_L=10.0, L=1.0, rand_shift=False, spec_transforms=None, frame_transforms=None, splits_path='./data', 
                meta_path='./data/info_r2plus1d_dim1024_15fps.json', frame_path='data/greatesthit/greatesthit_processed',
                p_outside_cond=0., p_audio_aug=0.5):
        super().__init__()
        self.split = split
        self.specs_dir = specs_dir
        self.spec_transforms = spec_transforms
        self.frame_transforms = frame_transforms
        self.splits_path = splits_path
        self.meta_path = meta_path
        self.frame_path = frame_path
        self.feat_len = feat_len
        self.feat_depth = feat_depth
        self.feat_crop_len = feat_crop_len
        self.spec_len = spec_len
        self.rand_shift = rand_shift
        self.L = L
        self.spec_take_first = int(math.ceil(860 * (vqgan_L / 10.) / 32) * 32)
        self.spec_take_first = 860 if self.spec_take_first > 860 else self.spec_take_first
        self.p_outside_cond = torch.tensor(p_outside_cond)

        greatesthit_meta = json.load(open(self.meta_path, 'r'))
        unique_classes = sorted(list(set(ht for ht in greatesthit_meta['hit_type'])))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}
        self.target2label = {target: label for label, target in self.label2target.items()}
        self.video_idx2label = {
            get_GH_data_identifier(greatesthit_meta['video_name'][i], greatesthit_meta['start_idx'][i]): 
            greatesthit_meta['hit_type'][i] for i in range(len(greatesthit_meta['video_name']))
        }
        self.available_video_hit = list(self.video_idx2label.keys())
        self.video_idx2path = {
            vh: os.path.join(self.specs_dir, 
                vh.replace('_', '_denoised_') + '_' + self.video_idx2label[vh].replace(' ', '_') +'_mel.npy')
            for vh in self.available_video_hit
        }
        for value in self.video_idx2path.values():
            assert os.path.exists(value)
        self.video_idx2idx = {
            get_GH_data_identifier(greatesthit_meta['video_name'][i], greatesthit_meta['start_idx'][i]):
            i for i in range(len(greatesthit_meta['video_name']))
        }

        split_clip_ids_path = os.path.join(splits_path, f'greatesthit_{split}.json')
        if not os.path.exists(split_clip_ids_path):
            self.make_split_files()
        clip_video_hit = json.load(open(split_clip_ids_path, 'r'))
        self.dataset = clip_video_hit
        spec_crop_len = self.spec_take_first if self.spec_take_first <= spec_crop_len else spec_crop_len
        self.spec_transforms = transforms.Compose([
            CropImage([mel_num, spec_crop_len], random_crop),
            # transforms.RandomApply([FrequencyMasking(freq_mask_param=20)], p=p_audio_aug),
            # transforms.RandomApply([TimeMasking(time_mask_param=int(32 * self.L))], p=p_audio_aug)
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

        clip_classes = [self.label2target[self.video_idx2label[vh]] for vh in clip_video_hit]
        class2count = collections.Counter(clip_classes)
        self.class_counts = torch.tensor([class2count[cls] for cls in range(len(class2count))])
        if self.L != 1.0:
            print(split, L)
            self.validate_data()
        self.video2indexes = {}
        for video_idx in self.dataset:
            video, start_idx = video_idx.split('_')
            if video not in self.video2indexes.keys():
                self.video2indexes[video] = []
            self.video2indexes[video].append(start_idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {}

        try:
            video_idx = self.dataset[idx]
            spec_path = self.video_idx2path[video_idx]
            spec = np.load(spec_path) # (80, 860)

            video, start_idx = video_idx.split('_')
            frame_path = os.path.join(self.frame_path, video, 'frames')
            start_frame_idx = non_negative(FPS * int(start_idx)/SR)
            end_frame_idx = non_negative(start_frame_idx + FPS * self.L)

            if self.rand_shift:
                shift = random.uniform(0, 0.5)
                spec_shift = int(shift * spec.shape[1] // 10)
                # Since only the first second is used
                spec = np.roll(spec, -spec_shift, 1)
                start_frame_idx += int(FPS * shift)
                end_frame_idx += int(FPS * shift)

            frames = [Image.open(os.path.join(
                frame_path, f'frame{i+1:0>6d}.jpg')).convert('RGB') for i in 
                range(start_frame_idx, end_frame_idx)]

            # Sample condition
            if torch.all(torch.bernoulli(self.p_outside_cond) == 1.):
                # Sample condition from outside video
                all_idx = set(list(range(len(self.dataset))))
                all_idx.remove(idx)
                cond_video_idx = self.dataset[sample(all_idx, k=1)[0]]
                cond_video, cond_start_idx = cond_video_idx.split('_')
            else:
                cond_video = video
                video_hits_idx = copy.copy(self.video2indexes[video])
                video_hits_idx.remove(start_idx)
                cond_start_idx = sample(video_hits_idx, k=1)[0]
                cond_video_idx = get_GH_data_identifier(cond_video, cond_start_idx)

            cond_spec_path = self.video_idx2path[cond_video_idx]
            cond_spec = np.load(cond_spec_path) # (80, 860)

            cond_video, cond_start_idx = cond_video_idx.split('_')
            cond_frame_path = os.path.join(self.frame_path, cond_video, 'frames')
            cond_start_frame_idx = non_negative(FPS * int(cond_start_idx)/SR)
            cond_end_frame_idx = non_negative(cond_start_frame_idx + FPS * self.L)

            if self.rand_shift:
                cond_shift = random.uniform(0, 0.5)
                cond_spec_shift = int(cond_shift * cond_spec.shape[1] // 10)
                # Since only the first second is used
                cond_spec = np.roll(cond_spec, -cond_spec_shift, 1)
                cond_start_frame_idx += int(FPS * cond_shift)
                cond_end_frame_idx += int(FPS * cond_shift)

            cond_frames = [Image.open(os.path.join(
                cond_frame_path, f'frame{i+1:0>6d}.jpg')).convert('RGB') for i in 
                range(cond_start_frame_idx, cond_end_frame_idx)]

            # concat spec outside dataload
            item['image'] = 2 * spec - 1 # (80, 860)
            item['cond_image'] = 2 * cond_spec - 1 # (80, 860)
            item['image'] = item['image'][:, :self.spec_take_first]
            item['cond_image'] = item['cond_image'][:, :self.spec_take_first]
            item['file_path_specs_'] = spec_path
            item['file_path_cond_specs_'] = cond_spec_path

            if self.frame_transforms is not None:
                cond_frames = self.frame_transforms(cond_frames)
                frames = self.frame_transforms(frames)

            item['feature'] = np.stack(cond_frames + frames, axis=0) # (30 * L, 112, 112, 3)
            item['file_path_feats_'] = (frame_path, start_frame_idx)
            item['file_path_cond_feats_'] = (cond_frame_path, cond_start_frame_idx)

            item['label'] = self.video_idx2label[video_idx]
            item['target'] = self.label2target[item['label']]

            if self.spec_transforms is not None:
                item = self.spec_transforms(item)
        except Exception:
            print(sys.exc_info()[2])
            print('!!!!!!!!!!!!!!!!!!!!', video_idx, cond_video_idx)
            print('!!!!!!!!!!!!!!!!!!!!', end_frame_idx, cond_end_frame_idx)
            exit(1)

        return item


    def validate_data(self):
        original_len = len(self.dataset)
        valid_dataset = []
        for video_idx in tqdm(self.dataset):
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
        split_clip_ids_path = os.path.join(self.splits_path, f'greatesthit_{self.split}_{self.L:.2f}.json')
        if not os.path.exists(split_clip_ids_path):
            with open(split_clip_ids_path, 'w') as f:
                json.dump(valid_dataset, f)


    def make_split_files(self, ratio=[0.85, 0.1, 0.05]):
        random.seed(1337)
        print(f'The split files do not exist @ {self.splits_path}. Calculating the new ones.')
        # The downloaded videos (some went missing on YouTube and no longer available)
        available_mel_paths = set(glob(os.path.join(self.specs_dir, '*_mel.npy')))
        self.available_video_hit = [vh for vh in self.available_video_hit if self.video_idx2path[vh] in available_mel_paths]

        all_video = list(self.video2indexes.keys())

        print(f'The number of clips available after download: {len(self.available_video_hit)}')
        print(f'The number of videos available after download: {len(all_video)}')

        available_idx = list(range(len(all_video)))
        random.shuffle(available_idx)
        assert sum(ratio) == 1.
        cut_train = int(ratio[0] * len(all_video))
        cut_test = cut_train + int(ratio[1] * len(all_video))

        train_idx = available_idx[:cut_train]
        test_idx = available_idx[cut_train:cut_test]
        valid_idx = available_idx[cut_test:]

        train_video = [all_video[i] for i in train_idx]
        test_video = [all_video[i] for i in test_idx]
        valid_video = [all_video[i] for i in valid_idx]

        train_video_hit = []
        for v in train_video:
            train_video_hit += [get_GH_data_identifier(v, hit_idx) for hit_idx in self.video2indexes[v]]
        test_video_hit = []
        for v in test_video:
            test_video_hit += [get_GH_data_identifier(v, hit_idx) for hit_idx in self.video2indexes[v]]
        valid_video_hit = []
        for v in valid_video:
            valid_video_hit += [get_GH_data_identifier(v, hit_idx) for hit_idx in self.video2indexes[v]]

        # mix train and valid for better validation loss
        mixed = train_video_hit + valid_video_hit
        random.shuffle(mixed)
        split = int(len(mixed) * ratio[0] / (ratio[0] + ratio[2]))
        train_video_hit = mixed[:split]
        valid_video_hit = mixed[split:]

        with open(os.path.join(self.splits_path, 'greatesthit_train.json'), 'w') as train_file,\
             open(os.path.join(self.splits_path, 'greatesthit_test.json'), 'w') as test_file,\
             open(os.path.join(self.splits_path, 'greatesthit_valid.json'), 'w') as valid_file:
            json.dump(train_video_hit, train_file)
            json.dump(test_video_hit, test_file)
            json.dump(valid_video_hit, valid_file)

        print(f'Put {len(train_idx)} clips to the train set and saved it to ./data/greatesthit_train.json')
        print(f'Put {len(test_idx)} clips to the test set and saved it to ./data/greatesthit_test.json')
        print(f'Put {len(valid_idx)} clips to the valid set and saved it to ./data/greatesthit_valid.json')


class CondGreatestHitSpecsCondOnImageTrain(CondGreatestHitSpecsCondOnImage):
    def __init__(self, dataset_cfg):
        train_transforms = transforms.Compose([
            Resize3D(256),
            RandomResizedCrop3D(224, scale=(0.5, 1.0)),
            RandomHorizontalFlip3D(),
            ColorJitter3D(brightness=0.1, saturation=0.1),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        super().__init__('train', frame_transforms=train_transforms, **dataset_cfg)

class CondGreatestHitSpecsCondOnImageValidation(CondGreatestHitSpecsCondOnImage):
    def __init__(self, dataset_cfg):
        valid_transforms = transforms.Compose([
            Resize3D(256),
            CenterCrop3D(224),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        super().__init__('val', frame_transforms=valid_transforms, **dataset_cfg)

class CondGreatestHitSpecsCondOnImageTest(CondGreatestHitSpecsCondOnImage):
    def __init__(self, dataset_cfg):
        test_transforms = transforms.Compose([
            Resize3D(256),
            CenterCrop3D(224),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        super().__init__('test', frame_transforms=test_transforms, **dataset_cfg)


class CondGreatestHitWaveCondOnImage(torch.utils.data.Dataset):

    def __init__(self, split, wav_dir, spec_len, random_crop, mel_num, spec_crop_len,
                L=2.0, frame_transforms=None, splits_path='./data',
                data_path='data/greatesthit/greatesthit-process-resized',
                p_outside_cond=0., p_audio_aug=0.5, rand_shift=True):
        super().__init__()
        self.split = split
        self.wav_dir = wav_dir
        self.frame_transforms = frame_transforms
        self.splits_path = splits_path
        self.data_path = data_path
        self.spec_len = spec_len
        self.L = L
        self.rand_shift = rand_shift
        self.p_outside_cond = torch.tensor(p_outside_cond)

        split_clip_ids_path = os.path.join(splits_path, f'greatesthit_{split}.json')
        if not os.path.exists(split_clip_ids_path):
            raise NotImplementedError()
        clip_video_hit = json.load(open(split_clip_ids_path, 'r'))

        video_name = list(set([vidx.split('_')[0] for vidx in clip_video_hit]))

        self.video_frame_cnt = {v: len(os.listdir(os.path.join(self.data_path, v, 'frames')))//2 for v in video_name}
        self.left_over = int(FPS * L + 1)
        self.video_audio_path = {v: os.path.join(self.data_path, v, f'audio/{v}_denoised_resampled.wav') for v in video_name}
        self.dataset = clip_video_hit

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

        self.wav_transforms = transforms.Compose([
            MakeMono(),
            Padding(target_len=int(SR * self.L)),
        ])
        if self.frame_transforms == None:
            self.frame_transforms = transforms.Compose([
                Resize3D(256),
                RandomResizedCrop3D(224, scale=(0.5, 1.0)),
                RandomHorizontalFlip3D(),
                ColorJitter3D(brightness=0.1, saturation=0.1),
                ToTensor3D(),
                Normalize3D(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {}
        video_idx = self.dataset[idx]
        video, start_idx = video_idx.split('_')
        start_idx = int(start_idx)
        frame_path = os.path.join(self.data_path, video, 'frames')
        start_frame_idx = non_negative(FPS * int(start_idx)/SR)
        if self.rand_shift:
            shift = random.uniform(-0.5, 0.5)
            start_frame_idx = non_negative(start_frame_idx + int(FPS * shift))
            start_idx = non_negative(start_idx + int(SR * shift))
        if start_frame_idx > self.video_frame_cnt[video] - self.left_over:
            start_frame_idx = self.video_frame_cnt[video] - self.left_over
            start_idx = non_negative(SR * (start_frame_idx / FPS))

        end_frame_idx = non_negative(start_frame_idx + FPS * self.L)

        # target
        wave_path = self.video_audio_path[video]
        frames = [Image.open(os.path.join(
            frame_path, f'frame{i+1:0>6d}_resize.jpg')).convert('RGB') for i in
            range(start_frame_idx, end_frame_idx)]
        wav, sr = soundfile.read(wave_path, frames=int(SR * self.L), start=start_idx)
        assert sr == SR
        wav = self.wav_transforms(wav)

        # cond
        if torch.all(torch.bernoulli(self.p_outside_cond) == 1.):
            all_idx = set(list(range(len(self.dataset))))
            all_idx.remove(idx)
            cond_video_idx = self.dataset[sample(all_idx, k=1)[0]]
            cond_video, cond_start_idx = cond_video_idx.split('_')
        else:
            cond_video = video
            video_hits_idx = copy.copy(self.video2indexes[video])
            if str(start_idx) in video_hits_idx:
                video_hits_idx.remove(str(start_idx))
            cond_start_idx = sample(video_hits_idx, k=1)[0]
            cond_video_idx = get_GH_data_identifier(cond_video, cond_start_idx)

        cond_video, cond_start_idx = cond_video_idx.split('_')
        cond_start_idx = int(cond_start_idx)
        cond_frame_path = os.path.join(self.data_path, cond_video, 'frames')
        cond_start_frame_idx = non_negative(FPS * int(cond_start_idx)/SR)
        cond_wave_path = self.video_audio_path[cond_video]

        if self.rand_shift:
            cond_shift = random.uniform(-0.5, 0.5)
            cond_start_frame_idx = non_negative(cond_start_frame_idx + int(FPS * cond_shift))
            cond_start_idx = non_negative(cond_start_idx + int(shift * SR))
        if cond_start_frame_idx > self.video_frame_cnt[cond_video] - self.left_over:
            cond_start_frame_idx = self.video_frame_cnt[cond_video] - self.left_over
            cond_start_idx = non_negative(SR * (cond_start_frame_idx / FPS))
        cond_end_frame_idx = non_negative(cond_start_frame_idx + FPS * self.L)

        cond_frames = [Image.open(os.path.join(
                cond_frame_path, f'frame{i+1:0>6d}_resize.jpg')).convert('RGB') for i in 
                range(cond_start_frame_idx, cond_end_frame_idx)]
        cond_wav, _ = soundfile.read(cond_wave_path, frames=int(SR * self.L), start=cond_start_idx)
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

    def validate_data(self):
        raise NotImplementedError()

    def make_split_files(self, ratio=[0.85, 0.1, 0.05]):
        random.seed(1337)
        print(f'The split files do not exist @ {self.splits_path}. Calculating the new ones.')

        all_video = sorted(os.listdir(self.data_path))
        print(f'The number of videos available after download: {len(all_video)}')

        available_idx = list(range(len(all_video)))
        random.shuffle(available_idx)
        assert sum(ratio) == 1.
        cut_train = int(ratio[0] * len(all_video))
        cut_test = cut_train + int(ratio[1] * len(all_video))

        train_idx = available_idx[:cut_train]
        test_idx = available_idx[cut_train:cut_test]
        valid_idx = available_idx[cut_test:]

        train_video = [all_video[i] for i in train_idx]
        test_video = [all_video[i] for i in test_idx]
        valid_video = [all_video[i] for i in valid_idx]

        with open(os.path.join(self.splits_path, 'greatesthit_video_train.json'), 'w') as train_file,\
             open(os.path.join(self.splits_path, 'greatesthit_video_test.json'), 'w') as test_file,\
             open(os.path.join(self.splits_path, 'greatesthit_video_valid.json'), 'w') as valid_file:
            json.dump(train_video, train_file)
            json.dump(test_video, test_file)
            json.dump(valid_video, valid_file)

        print(f'Put {len(train_idx)} videos to the train set and saved it to ./data/greatesthit_video_train.json')
        print(f'Put {len(test_idx)} videos to the test set and saved it to ./data/greatesthit_video_test.json')
        print(f'Put {len(valid_idx)} videos to the valid set and saved it to ./data/greatesthit_video_valid.json')


class CondGreatestHitWaveCondOnImageTrain(CondGreatestHitWaveCondOnImage):
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

class CondGreatestHitWaveCondOnImageValidation(CondGreatestHitWaveCondOnImage):
    def __init__(self, dataset_cfg):
        valid_transforms = transforms.Compose([
            Resize3D(128),
            CenterCrop3D(112),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        super().__init__('val', frame_transforms=valid_transforms, **dataset_cfg)

class CondGreatestHitWaveCondOnImageTest(CondGreatestHitWaveCondOnImage):
    def __init__(self, dataset_cfg):
        test_transforms = transforms.Compose([
            Resize3D(128),
            CenterCrop3D(112),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        super().__init__('test', frame_transforms=test_transforms, **dataset_cfg)



class GreatestHitWaveCondOnImage(torch.utils.data.Dataset):

    def __init__(self, split, wav_dir, spec_len, random_crop, mel_num, spec_crop_len,
                L=2.0, frame_transforms=None, splits_path='./data',
                data_path='data/greatesthit/greatesthit-process-resized',
                p_outside_cond=0., p_audio_aug=0.5, rand_shift=True):
        super().__init__()
        self.split = split
        self.wav_dir = wav_dir
        self.frame_transforms = frame_transforms
        self.splits_path = splits_path
        self.data_path = data_path
        self.spec_len = spec_len
        self.L = L
        self.rand_shift = rand_shift
        self.p_outside_cond = torch.tensor(p_outside_cond)

        split_clip_ids_path = os.path.join(splits_path, f'greatesthit_{split}.json')
        if not os.path.exists(split_clip_ids_path):
            raise NotImplementedError()
        clip_video_hit = json.load(open(split_clip_ids_path, 'r'))

        video_name = list(set([vidx.split('_')[0] for vidx in clip_video_hit]))

        self.video_frame_cnt = {v: len(os.listdir(os.path.join(self.data_path, v, 'frames')))//2 for v in video_name}
        self.left_over = int(FPS * L + 1)
        self.video_audio_path = {v: os.path.join(self.data_path, v, f'audio/{v}_denoised_resampled.wav') for v in video_name}
        self.dataset = clip_video_hit

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

        self.wav_transforms = transforms.Compose([
            MakeMono(),
            Padding(target_len=int(SR * self.L)),
        ])
        if self.frame_transforms == None:
            self.frame_transforms = transforms.Compose([
                Resize3D(256),
                RandomResizedCrop3D(224, scale=(0.5, 1.0)),
                RandomHorizontalFlip3D(),
                ColorJitter3D(brightness=0.1, saturation=0.1),
                ToTensor3D(),
                Normalize3D(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {}
        video_idx = self.dataset[idx]
        video, start_idx = video_idx.split('_')
        start_idx = int(start_idx)
        frame_path = os.path.join(self.data_path, video, 'frames')
        start_frame_idx = non_negative(FPS * int(start_idx)/SR)
        if self.rand_shift:
            shift = random.uniform(-0.5, 0.5)
            start_frame_idx = non_negative(start_frame_idx + int(FPS * shift))
            start_idx = non_negative(start_idx + int(SR * shift))
        if start_frame_idx > self.video_frame_cnt[video] - self.left_over:
            start_frame_idx = self.video_frame_cnt[video] - self.left_over
            start_idx = non_negative(SR * (start_frame_idx / FPS))

        end_frame_idx = non_negative(start_frame_idx + FPS * self.L)

        # target
        wave_path = self.video_audio_path[video]
        frames = [Image.open(os.path.join(
            frame_path, f'frame{i+1:0>6d}_resize.jpg')).convert('RGB') for i in
            range(start_frame_idx, end_frame_idx)]
        wav, sr = soundfile.read(wave_path, frames=int(SR * self.L), start=start_idx)
        assert sr == SR
        wav = self.wav_transforms(wav)

        item['image'] = wav # (44100,)
        item['file_path_wav_'] = wave_path

        if self.frame_transforms is not None:
            frames = self.frame_transforms(frames)

        item['feature'] = torch.stack(frames, dim=0) # (15 * L, 112, 112, 3)
        item['file_path_feats_'] = (frame_path, start_idx)

        item['label'] = 'None'
        item['target'] = 'None'

        return item

    def validate_data(self):
        raise NotImplementedError()

    def make_split_files(self, ratio=[0.85, 0.1, 0.05]):
        random.seed(1337)
        print(f'The split files do not exist @ {self.splits_path}. Calculating the new ones.')

        all_video = sorted(os.listdir(self.data_path))
        print(f'The number of videos available after download: {len(all_video)}')

        available_idx = list(range(len(all_video)))
        random.shuffle(available_idx)
        assert sum(ratio) == 1.
        cut_train = int(ratio[0] * len(all_video))
        cut_test = cut_train + int(ratio[1] * len(all_video))

        train_idx = available_idx[:cut_train]
        test_idx = available_idx[cut_train:cut_test]
        valid_idx = available_idx[cut_test:]

        train_video = [all_video[i] for i in train_idx]
        test_video = [all_video[i] for i in test_idx]
        valid_video = [all_video[i] for i in valid_idx]

        with open(os.path.join(self.splits_path, 'greatesthit_video_train.json'), 'w') as train_file,\
             open(os.path.join(self.splits_path, 'greatesthit_video_test.json'), 'w') as test_file,\
             open(os.path.join(self.splits_path, 'greatesthit_video_valid.json'), 'w') as valid_file:
            json.dump(train_video, train_file)
            json.dump(test_video, test_file)
            json.dump(valid_video, valid_file)

        print(f'Put {len(train_idx)} videos to the train set and saved it to ./data/greatesthit_video_train.json')
        print(f'Put {len(test_idx)} videos to the test set and saved it to ./data/greatesthit_video_test.json')
        print(f'Put {len(valid_idx)} videos to the valid set and saved it to ./data/greatesthit_video_valid.json')


class GreatestHitWaveCondOnImageTrain(GreatestHitWaveCondOnImage):
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

class GreatestHitWaveCondOnImageValidation(GreatestHitWaveCondOnImage):
    def __init__(self, dataset_cfg):
        valid_transforms = transforms.Compose([
            Resize3D(128),
            CenterCrop3D(112),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        super().__init__('val', frame_transforms=valid_transforms, **dataset_cfg)

class GreatestHitWaveCondOnImageTest(GreatestHitWaveCondOnImage):
    def __init__(self, dataset_cfg):
        test_transforms = transforms.Compose([
            Resize3D(128),
            CenterCrop3D(112),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        super().__init__('test', frame_transforms=test_transforms, **dataset_cfg)


def draw_spec(spec, dest, cmap='magma'):
    plt.imshow(spec, cmap=cmap, origin='lower')
    plt.axis('off')
    plt.savefig(dest, bbox_inches='tight', pad_inches=0., dpi=300)
    plt.close()

if __name__ == '__main__':
    import sys

    from omegaconf import OmegaConf

    # cfg = OmegaConf.load('configs/greatesthit_transformer_with_vNet_randshift_2s_GH_vqgan_no_earlystop.yaml')
    cfg = OmegaConf.load('configs/greatesthit_codebook.yaml')
    data = instantiate_from_config(cfg.data)
    data.prepare_data()
    data.setup()
    print(len(data.datasets['train']))
    print(data.datasets['train'][24])

