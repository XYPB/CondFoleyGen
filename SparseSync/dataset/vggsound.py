import csv
import logging
import os
import random
import sys
from collections import Counter
from glob import glob
from pathlib import Path

import torch
import torchvision

sys.path.insert(0, '.')  # nopep8
from dataset.dataset_utils import (get_audio_stream, get_fixed_offsets, get_video_and_audio)

logger = logging.getLogger(f'main.{__name__}')


class VGGSound(torch.utils.data.Dataset):

    def __init__(self,
                 split,
                 vids_dir,
                 transforms=None,
                 to_filter_bad_examples=True,
                 splits_path='./data',
                 meta_path='./data/vggsound.csv',
                 seed=1337,
                 load_fixed_offsets_on_test=True,
                 vis_load_backend='read_video',
                 size_ratio=None):
        super().__init__()
        self.max_clip_len_sec = None
        self.split = split
        self.vids_dir = vids_dir
        self.transforms = transforms
        self.to_filter_bad_examples = to_filter_bad_examples
        self.splits_path = splits_path
        self.meta_path = meta_path
        self.seed = seed
        self.load_fixed_offsets_on_test = load_fixed_offsets_on_test
        self.vis_load_backend = vis_load_backend
        self.size_ratio = size_ratio

        vggsound_meta = list(csv.reader(open(meta_path), quotechar='"'))

        # filter "bad" examples
        if to_filter_bad_examples:
            vggsound_meta = self.filter_bad_examples(vggsound_meta)

        unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}
        self.target2label = {target: label for label, target in self.label2target.items()}
        self.video2target = {row[0]: self.label2target[row[2]] for row in vggsound_meta}

        split_clip_ids_path = os.path.join(splits_path, f'vggsound_{split}.txt')
        if not os.path.exists(split_clip_ids_path):
            self.make_split_files()
        # the ugly string converts ['AdfsGsfII2yQ', '1'] into `AdfsGsfII2yQ_1000_11000`
        meta_available = set([f'{r[0]}_{int(r[1])*1000}_{(int(r[1])+10)*1000}' for r in vggsound_meta])
        within_split = set(open(split_clip_ids_path).read().splitlines())
        clip_paths = [os.path.join(vids_dir, v + '.mp4') for v in meta_available.intersection(within_split)]
        clip_paths = sorted(clip_paths)

        # loading the fixed offsets. COMMENT THIS IF YOU DON'T HAVE A FILE YET
        if load_fixed_offsets_on_test and split in ['valid', 'test']:
            logger.info(f'Using fixed offset for {split}')
            self.vid2offset_params = get_fixed_offsets(transforms, split, splits_path, 'vggsound')

        # making sure that all classes have at least one example
        counter = Counter([self.video2target[Path(p).stem[:11]] for p in clip_paths])
        assert all(counter[c] > 0 for c in self.target2label.keys()), \
            f'Some classes have 0 count: {dict(counter)}'

        self.dataset = clip_paths
        if size_ratio is not None and 0.0 < size_ratio < 1.0:
            cut_off = int(len(self.dataset) * size_ratio)
            self.dataset = self.dataset[:cut_off]

        logger.info(f'{split} has {len(self.dataset)} items')

    def filter_bad_examples(self, vggsound_meta):
        bad = set()
        base_path = Path('./data/filtered_examples_vggsound')
        lists = [open(p).read().splitlines() for p in sorted(glob(str(base_path / '*.txt')))]
        for s in lists:
            bad = bad.union(s)
        # the ugly string converts '---g-f_I2yQ', '1' into `---g-f_I2yQ_1000_11000`
        vggsound_meta = [r for r in vggsound_meta if f'{r[0]}_{int(r[1])*1000}_{(int(r[1])+10)*1000}' not in bad]
        return vggsound_meta

    def __getitem__(self, index):
        path = self.dataset[index]
        rgb, audio, meta = self.load_media(path)
        item = self.make_datapoint(path, rgb, audio, meta)
        if self.transforms is not None:
            item = self.transforms(item)
        return item

    def make_datapoint(self, path, rgb, audio, meta):
        # (Tv, 3, H, W) in [0, 225], (Ta, C) in [-1, 1]
        target = self.video2target[Path(path).stem[:11]]
        item = {
            'video': rgb,
            'audio': audio,
            'meta': meta,
            'path': str(path),
            'targets': {'vggsound_target': target, 'vggsound_label': self.target2label[target]},
            'split': self.split,
        }

        # loading the fixed offsets. COMMENT THIS IF YOU DON'T HAVE A FILE YET
        if self.load_fixed_offsets_on_test and self.split in ['valid', 'test']:
            item['targets']['offset_sec'] = self.vid2offset_params[Path(path).stem]['offset_sec']
            item['targets']['v_start_i_sec'] = self.vid2offset_params[Path(path).stem]['v_start_i_sec']

        return item

    def load_media(self, path):
        rgb, audio, meta = get_video_and_audio(path, get_meta=True, max_clip_len_sec=self.max_clip_len_sec)
        return rgb, audio, meta

    def make_split_files(self):
        if self.to_filter_bad_examples:
            logger.warning('`to_filter_bad_examples` is True. `make_split_files` expects otherwise')

        logger.info(f'The split files do not exist @ {self.splits_path}. Calculating the new ones.')
        # The downloaded videos (some went missing on YouTube and no longer available)
        available_vid_paths = sorted(glob(os.path.join(self.vids_dir, '*.mp4')))
        logger.info(f'The number of clips available after download: {len(available_vid_paths)}')

        # original (full) train and test sets
        vggsound_meta = list(csv.reader(open(self.meta_path), quotechar='"'))
        train_vids = {row[0] for row in vggsound_meta if row[3] == 'train'}
        test_vids = {row[0] for row in vggsound_meta if row[3] == 'test'}

        # # the cleaned test set
        # vggsound_meta_test_v2 = list(csv.reader(open(self.meta_path_clean_test), quotechar='"'))
        logger.info(f'The number of videos in vggsound train set: {len(train_vids)}')
        logger.info(f'The number of videos in vggsound test set: {len(test_vids)}')

        # class counts in test set. We would like to have the same distribution in valid
        unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
        label2target = {label: target for target, label in enumerate(unique_classes)}
        video2target = {row[0]: label2target[row[2]] for row in vggsound_meta}
        test_vid_classes = [video2target[vid] for vid in test_vids]
        test_target2count = Counter(test_vid_classes)

        # now given the counts from test set, sample the same count for validation and the rest leave in train
        train_vids_wo_valid, valid_vids = set(), set()
        for target, label in enumerate(label2target.keys()):
            class_train_vids = [vid for vid in sorted(list(train_vids)) if video2target[vid] == target]
            random.Random(self.seed).shuffle(class_train_vids)
            count = test_target2count[target]
            valid_vids.update(class_train_vids[:count])
            train_vids_wo_valid.update(class_train_vids[count:])

        # make file with a list of available test videos (each video should contain timestamps as well)
        train_i = valid_i = test_i = 0
        with open(os.path.join(self.splits_path, 'vggsound_train.txt'), 'w') as train_file, \
             open(os.path.join(self.splits_path, 'vggsound_valid.txt'), 'w') as valid_file, \
             open(os.path.join(self.splits_path, 'vggsound_test.txt'), 'w') as test_file:
            #  open(os.path.join(self.splits_path, 'vggsound_test_v2.txt'), 'w') as test_file_v2:
            for path in available_vid_paths:
                path = path.replace('.mp4', '')
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
                # else:
                #     raise Exception(f'Clip {vid_name} is neither in train, valid nor test. Strange.')

        logger.info(f'Put {train_i} clips to the train set and saved it to ./data/vggsound_train.txt')
        logger.info(f'Put {valid_i} clips to the valid set and saved it to ./data/vggsound_valid.txt')
        logger.info(f'Put {test_i} clips to the test set and saved it to ./data/vggsound_test.txt')

    def __len__(self):
        return len(self.dataset)

class VGGSoundAudioOnly(VGGSound):

    def __init__(self, split, vids_dir, transforms=None, to_filter_bad_examples=False,
                 splits_path='./data', meta_path='./data/vggsound.csv', seed=1337,
                 load_fixed_offsets_on_test=False, size_ratio=None):
        super().__init__(split, vids_dir, transforms, to_filter_bad_examples,
                         splits_path, meta_path, seed, load_fixed_offsets_on_test, size_ratio)

    def __getitem__(self, index):
        path = self.dataset[index]
        # (Ta, C) in [-1, 1]
        audio, meta = get_audio_stream(path, get_meta=True)

        target = self.video2target[Path(path).stem[:11]]
        item = {
            'audio': audio,
            'meta': meta,
            'path': path,
            'target': target,
            'label': self.target2label[target],
            'split': self.split,
        }

        if self.transforms is not None:
            item = self.transforms(item)

        return item

class VGGSoundSparse(VGGSound):
    '''
        The same as VGGSound, except the list of videos is filtered for sparse sounds
    '''

    def __init__(self, split, vids_dir, transforms=None, to_filter_bad_examples=True,
                 splits_path='./data', meta_path='./data/vggsound.csv',
                 sparse_meta_path='./data/sparse_classes.csv', seed=1337, load_fixed_offsets_on_test=True,
                 vis_load_backend='read_video', size_ratio=None):
        super().__init__(split, vids_dir, transforms, to_filter_bad_examples, splits_path, meta_path, seed,
                         load_fixed_offsets_on_test, vis_load_backend, size_ratio)
        self.sparse_meta_path = sparse_meta_path
        sparse_meta = list(csv.reader(open(sparse_meta_path), quotechar='"', delimiter='\t'))
        sparse_classes = set([row[0] for row in sparse_meta if row[1] == 'y'])
        label2new_target = {label: target for target, label in enumerate(sorted(list(sparse_classes)))}
        new_target2label = {target: label for label, target in label2new_target.items()}

        sparse_dataset = []
        video2new_target = {}
        for path in self.dataset:
            vid_id = Path(path).stem[:11]
            vid_target = self.video2target[vid_id]
            vid_label = self.target2label[vid_target]
            if vid_label in sparse_classes:
                sparse_dataset.append(path)
                video2new_target[vid_id] = label2new_target[vid_label]

        self.dataset = sparse_dataset

        # redefining the label <-> target variable
        self.label2target = label2new_target
        self.target2label = new_target2label
        self.video2target = video2new_target

        counter = Counter([self.video2target[Path(p).stem[:11]] for p in self.dataset])
        assert len(self.dataset) < 1000 or all(counter[c] > 0 for c in self.target2label.keys()), \
            f'Some classes have 0 count: {dict(counter)}'

class VGGSoundSparsePicked(VGGSoundSparse):

    def __init__(self, split, vids_dir, transforms=None, to_filter_bad_examples=True,
                 splits_path='./data', meta_path='./data/vggsound.csv',
                 sparse_meta_path='./data/picked_sparse_classes.csv', seed=1337,
                 load_fixed_offsets_on_test=True, vis_load_backend='read_video', size_ratio=None):
        super().__init__(split, vids_dir, transforms, to_filter_bad_examples, splits_path,
                         meta_path, sparse_meta_path, seed, load_fixed_offsets_on_test, vis_load_backend,
                         size_ratio)
        logger.info(f'{split} has {len(self.dataset)} items')


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from scripts.train_utils import get_transforms
    from utils.utils import cfg_sanity_check_and_patch
    cfg = OmegaConf.load('./configs/transformer.yaml')
    vis_load_backend = 'read_video'

    transforms = get_transforms(cfg)

    vids_path = 'PLACEHOLDER'

    # cfg.data.dataset.params.size_ratio = 0.1

    cfg_sanity_check_and_patch(cfg)

    datasets = {
        'train': VGGSound('train', vids_path, transforms['train'], vis_load_backend=vis_load_backend,
                          size_ratio=cfg.data.dataset.params.size_ratio),
        'valid': VGGSound('valid', vids_path, transforms['test'], vis_load_backend=vis_load_backend),
        'test': VGGSound('test', vids_path, transforms['test'], vis_load_backend=vis_load_backend),
    }
    for phase in ['train', 'valid', 'test']:
        print(phase, len(datasets[phase]))

    print(datasets['train'][0]['audio'].shape, datasets['train'][0]['video'].shape)
    print(datasets['train'][0]['meta'])
    print(datasets['valid'][0]['audio'].shape, datasets['valid'][0]['video'].shape)
    print(datasets['valid'][0]['meta'])
    print(datasets['test'][0]['audio'].shape, datasets['test'][0]['video'].shape)
    print(datasets['test'][0]['meta'])

    # for i in range(300, 1000):
    #     datasets['train'][i]['path']
    #     print(datasets['train'][0]['audio'].shape, datasets['train'][0]['video'].shape)
    #     print(datasets['train'][0]['meta'])

    transforms_dummy = torchvision.transforms.transforms.Compose([
        lambda x: x,
    ])
    datasets = {
        'train': VGGSoundAudioOnly('train', vids_path, transforms_dummy, to_filter_bad_examples=True),
        'valid': VGGSoundAudioOnly('valid', vids_path, transforms_dummy, to_filter_bad_examples=True),
        'test': VGGSoundAudioOnly('test', vids_path, transforms_dummy, to_filter_bad_examples=True),
    }
    for phase in ['train', 'valid', 'test']:
        print(phase, len(datasets[phase]))

    print(datasets['train'][0]['audio'].shape)
    print(datasets['train'][0]['meta'])
    print(datasets['train'][0])
    print(datasets['train'][0]['target'], datasets['train'][0]['label'])
    print(datasets['valid'][0]['audio'].shape)
    print(datasets['valid'][0]['meta'])
    print(datasets['valid'][0]['target'], datasets['valid'][0]['label'])
    print(datasets['test'][0]['audio'].shape)
    print(datasets['test'][0]['meta'])
    print(datasets['test'][0]['target'], datasets['test'][0]['label'])

    datasets = {
        'train': VGGSoundSparse('train', vids_path, transforms['train']),
        'valid': VGGSoundSparse('valid', vids_path, transforms['test']),
        'test': VGGSoundSparse('test', vids_path, transforms['test']),
    }
    for phase in ['train', 'valid', 'test']:
        print(phase, len(datasets[phase]))

    print(datasets['train'][0]['audio'].shape, datasets['train'][0]['video'].shape)
    print(datasets['train'][0]['meta'])
    print(datasets['valid'][0]['audio'].shape, datasets['valid'][0]['video'].shape)
    print(datasets['valid'][0]['meta'])
    print(datasets['test'][0]['audio'].shape, datasets['test'][0]['video'].shape)
    print(datasets['test'][0]['meta'])
