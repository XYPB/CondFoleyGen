import collections
import csv
import logging
import os
import random
import math
import json
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torchvision

logger = logging.getLogger(f'main.{__name__}')


class VGGSound(torch.utils.data.Dataset):

    def __init__(self, split, specs_dir, transforms=None, splits_path='./data', meta_path='./data/vggsound.csv'):
        super().__init__()
        self.split = split
        self.specs_dir = specs_dir
        self.transforms = transforms
        self.splits_path = splits_path
        self.meta_path = meta_path

        vggsound_meta = list(csv.reader(open(meta_path), quotechar='"'))
        unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}
        self.target2label = {target: label for label, target in self.label2target.items()}
        self.video2target = {row[0]: self.label2target[row[2]] for row in vggsound_meta}

        split_clip_ids_path = os.path.join(splits_path, f'vggsound_{split}_partial.txt')
        print('&&&&&&&&&&&&&&&&', split_clip_ids_path)
        if not os.path.exists(split_clip_ids_path):
            self.make_split_files()
        clip_ids_with_timestamp = open(split_clip_ids_path).read().splitlines()
        clip_paths = [os.path.join(specs_dir, v + '_mel.npy') for v in clip_ids_with_timestamp]
        self.dataset = clip_paths
        # self.dataset = clip_paths[:10000]  # overfit one batch

        # 'zyTX_1BXKDE_16000_26000'[:11] -> 'zyTX_1BXKDE'
        vid_classes = [self.video2target[Path(path).stem[:11]] for path in self.dataset]
        class2count = collections.Counter(vid_classes)
        self.class_counts = torch.tensor([class2count[cls] for cls in range(len(class2count))])
        # self.sample_weights = [len(self.dataset) / class2count[self.video2target[Path(path).stem[:11]]] for path in self.dataset]

    def __getitem__(self, idx):
        item = {}

        spec_path = self.dataset[idx]
        # 'zyTX_1BXKDE_16000_26000' -> 'zyTX_1BXKDE'
        video_name = Path(spec_path).stem[:11]

        item['input'] = np.load(spec_path)
        item['input_path'] = spec_path

        # if self.split in ['train', 'valid']:
        item['target'] = self.video2target[video_name]
        item['label'] = self.target2label[item['target']]

        if self.transforms is not None:
            item = self.transforms(item)

        return item

    def __len__(self):
        return len(self.dataset)

    def make_split_files(self):
        random.seed(1337)
        logger.info(f'The split files do not exist @ {self.splits_path}. Calculating the new ones.')
        # The downloaded videos (some went missing on YouTube and no longer available)
        available_vid_paths = sorted(glob(os.path.join(self.specs_dir, '*_mel.npy')))
        logger.info(f'The number of clips available after download: {len(available_vid_paths)}')

        # original (full) train and test sets
        vggsound_meta = list(csv.reader(open(self.meta_path), quotechar='"'))
        train_vids = {row[0] for row in vggsound_meta if row[3] == 'train'}
        test_vids = {row[0] for row in vggsound_meta if row[3] == 'test'}
        logger.info(f'The number of videos in vggsound train set: {len(train_vids)}')
        logger.info(f'The number of videos in vggsound test set: {len(test_vids)}')

        # class counts in test set. We would like to have the same distribution in valid
        unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
        label2target = {label: target for target, label in enumerate(unique_classes)}
        video2target = {row[0]: label2target[row[2]] for row in vggsound_meta}
        test_vid_classes = [video2target[vid] for vid in test_vids]
        test_target2count = collections.Counter(test_vid_classes)

        # now given the counts from test set, sample the same count for validation and the rest leave in train
        train_vids_wo_valid, valid_vids = set(), set()
        for target, label in enumerate(label2target.keys()):
            class_train_vids = [vid for vid in train_vids if video2target[vid] == target]
            random.shuffle(class_train_vids)
            count = test_target2count[target]
            valid_vids.update(class_train_vids[:count])
            train_vids_wo_valid.update(class_train_vids[count:])

        # make file with a list of available test videos (each video should contain timestamps as well)
        train_i = valid_i = test_i = 0
        with open(os.path.join(self.splits_path, 'vggsound_train.txt'), 'w') as train_file, \
             open(os.path.join(self.splits_path, 'vggsound_valid.txt'), 'w') as valid_file, \
             open(os.path.join(self.splits_path, 'vggsound_test.txt'), 'w') as test_file:
            for path in available_vid_paths:
                path = path.replace('_mel.npy', '')
                vid_name = Path(path).name
                # 'zyTX_1BXKDE_16000_26000'[:11] -> 'zyTX_1BXKDE'
                if vid_name[:11] in train_vids_wo_valid:
                    train_file.write(vid_name + '\n')
                    train_i += 1
                elif vid_name[:11] in valid_vids:
                    valid_file.write(vid_name + '\n')
                    valid_i += 1
                elif vid_name[:11] in test_vids:
                    test_file.write(vid_name + '\n')
                    test_i += 1
                else:
                    raise Exception(f'Clip {vid_name} is neither in train, valid nor test. Strange.')

        logger.info(f'Put {train_i} clips to the train set and saved it to ./data/vggsound_train.txt')
        logger.info(f'Put {valid_i} clips to the valid set and saved it to ./data/vggsound_valid.txt')
        logger.info(f'Put {test_i} clips to the test set and saved it to ./data/vggsound_test.txt')


def get_GH_data_identifier(video_name, start_idx, split='_'):
    if isinstance(start_idx, str):
        return video_name + split + start_idx
    elif isinstance(start_idx, int):
        return video_name + split + str(start_idx)
    else:
        raise NotImplementedError


class GreatestHit(torch.utils.data.Dataset):

    def __init__(self, split, spec_dir_path, spec_transform=None, L=2.0, action_only=False,
                material_only=False, splits_path='/home/duyxxd/SpecVQGAN/data', 
                meta_path='/home/duyxxd/SpecVQGAN/data/info_r2plus1d_dim1024_15fps.json'):
        super().__init__()
        self.split = split
        self.specs_dir = spec_dir_path
        self.splits_path = splits_path
        self.meta_path = meta_path
        self.spec_transform = spec_transform
        self.L = L
        self.spec_take_first = int(math.ceil(860 * (L / 10.) / 32) * 32)
        self.spec_take_first = 860 if self.spec_take_first > 860 else self.spec_take_first
        self.spec_take_first = 173

        greatesthit_meta = json.load(open(self.meta_path, 'r'))
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

        split_clip_ids_path = os.path.join(splits_path, f'greatesthit_{split}_2.00_single_type_only.json')
        if not os.path.exists(split_clip_ids_path):
            raise NotImplementedError()
        clip_video_hit = json.load(open(split_clip_ids_path, 'r'))
        self.dataset = list(clip_video_hit.keys())
        if action_only:
            self.video_idx2label = {k: v.split(' ')[1] for k, v in clip_video_hit.items()}
        elif material_only:
            self.video_idx2label = {k: v.split(' ')[0] for k, v in clip_video_hit.items()}
        else:
            self.video_idx2label = clip_video_hit


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

        vid_classes = list(self.video_idx2label.values())
        unique_classes = sorted(list(set(vid_classes)))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}
        if action_only:
            label2target_fix = {'hit': 0, 'scratch': 1}
        elif material_only:
            label2target_fix = {'carpet': 0, 'ceramic': 1, 'cloth': 2, 'dirt': 3, 'drywall': 4, 'glass': 5, 'grass': 6, 'gravel': 7, 'leaf': 8, 'metal': 9, 'paper': 10, 'plastic': 11, 'plastic-bag': 12, 'rock': 13, 'tile': 14, 'water': 15, 'wood': 16}
        else:
            label2target_fix = {'carpet hit': 0, 'carpet scratch': 1, 'ceramic hit': 2, 'ceramic scratch': 3, 'cloth hit': 4, 'cloth scratch': 5, 'dirt hit': 6, 'dirt scratch': 7, 'drywall hit': 8, 'drywall scratch': 9, 'glass hit': 10, 'glass scratch': 11, 'grass hit': 12, 'grass scratch': 13, 'gravel hit': 14, 'gravel scratch': 15, 'leaf hit': 16, 'leaf scratch': 17, 'metal hit': 18, 'metal scratch': 19, 'paper hit': 20, 'paper scratch': 21, 'plastic hit': 22, 'plastic scratch': 23, 'plastic-bag hit': 24, 'plastic-bag scratch': 25, 'rock hit': 26, 'rock scratch': 27, 'tile hit': 28, 'tile scratch': 29, 'water hit': 30, 'water scratch': 31, 'wood hit': 32, 'wood scratch': 33}
        for k in self.label2target.keys():
            assert k in label2target_fix.keys()
        self.label2target = label2target_fix
        self.target2label = {target: label for label, target in self.label2target.items()}
        class2count = collections.Counter(vid_classes)
        self.class_counts = torch.tensor([class2count[cls] for cls in range(len(class2count))])
        print(self.label2target)
        print(len(vid_classes), len(class2count), class2count)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {}

        video_idx = self.dataset[idx]
        spec_path = self.video_idx2path[video_idx]
        spec = np.load(spec_path) # (80, 860)

        # concat spec outside dataload
        item['input'] = 2 * spec - 1 # (80, 860)
        item['input'] = item['input'][:, :self.spec_take_first] # (80, 173) (since 2sec audio can only generate 173)
        item['file_path'] = spec_path

        item['label'] = self.video_idx2label[video_idx]
        item['target'] = self.label2target[item['label']]

        if self.spec_transform is not None:
            item = self.spec_transform(item)

        return item



class AMT_test(torch.utils.data.Dataset):

    def __init__(self, spec_dir_path, spec_transform=None, action_only=False, material_only=False):
        super().__init__()
        self.specs_dir = spec_dir_path
        self.spec_transform = spec_transform
        self.spec_take_first = 173

        self.dataset = sorted([os.path.join(self.specs_dir, f) for f in os.listdir(self.specs_dir)])
        if action_only:
            self.label2target = {'hit': 0, 'scratch': 1}
        elif material_only:
            self.label2target = {'carpet': 0, 'ceramic': 1, 'cloth': 2, 'dirt': 3, 'drywall': 4, 'glass': 5, 'grass': 6, 'gravel': 7, 'leaf': 8, 'metal': 9, 'paper': 10, 'plastic': 11, 'plastic-bag': 12, 'rock': 13, 'tile': 14, 'water': 15, 'wood': 16}
        else:
            self.label2target = {'carpet hit': 0, 'carpet scratch': 1, 'ceramic hit': 2, 'ceramic scratch': 3, 'cloth hit': 4, 'cloth scratch': 5, 'dirt hit': 6, 'dirt scratch': 7, 'drywall hit': 8, 'drywall scratch': 9, 'glass hit': 10, 'glass scratch': 11, 'grass hit': 12, 'grass scratch': 13, 'gravel hit': 14, 'gravel scratch': 15, 'leaf hit': 16, 'leaf scratch': 17, 'metal hit': 18, 'metal scratch': 19, 'paper hit': 20, 'paper scratch': 21, 'plastic hit': 22, 'plastic scratch': 23, 'plastic-bag hit': 24, 'plastic-bag scratch': 25, 'rock hit': 26, 'rock scratch': 27, 'tile hit': 28, 'tile scratch': 29, 'water hit': 30, 'water scratch': 31, 'wood hit': 32, 'wood scratch': 33}
        self.target2label = {v: k for k, v in self.label2target.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {}

        spec_path = self.dataset[idx]
        spec = np.load(spec_path) # (80, 860)

        # concat spec outside dataload
        item['input'] = 2 * spec - 1 # (80, 860)
        item['input'] = item['input'][:, :self.spec_take_first] # (80, 173) (since 2sec audio can only generate 173)
        item['file_path'] = spec_path

        if self.spec_transform is not None:
            item = self.spec_transform(item)

        return item


if __name__ == '__main__':
    from transforms import Crop, StandardNormalizeAudio, ToTensor
    specs_path = '/home/nvme/data/vggsound/features/melspec_10s_22050hz/'

    transforms = torchvision.transforms.transforms.Compose([
        StandardNormalizeAudio(specs_path),
        ToTensor(),
        Crop([80, 848]),
    ])

    datasets = {
        'train': VGGSound('train', specs_path, transforms),
        'valid': VGGSound('valid', specs_path, transforms),
        'test': VGGSound('test', specs_path, transforms),
    }

    print(datasets['train'][0])
    print(datasets['valid'][0])
    print(datasets['test'][0])

    print(datasets['train'].class_counts)
    print(datasets['valid'].class_counts)
    print(datasets['test'].class_counts)
