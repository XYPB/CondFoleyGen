import pickle
import logging
import os
import random
import sys
from glob import glob
from pathlib import Path

import torch

sys.path.insert(0, '.')  # nopep8
from dataset.dataset_utils import (get_fixed_offsets, get_video_and_audio)

logger = logging.getLogger(f'main.{__name__}')


class LRS3(torch.utils.data.Dataset):

    def __init__(self,
                 split,
                 vids_dir,
                 transforms=None,
                 splits_path='./data',
                 seed=1337,
                 load_fixed_offsets_on_test=True,
                 vis_load_backend='VideoReader',
                 size_ratio=None):
        super().__init__()
        self.max_clip_len_sec = 11
        logger.info(f'During IO, the length of clips is limited to {self.max_clip_len_sec} sec')
        logger.info(f'During IO, the length of clips is limited to {self.max_clip_len_sec} sec')
        self.split = split
        self.vids_dir = vids_dir
        self.transforms = transforms
        self.splits_path = splits_path
        self.seed = seed
        self.load_fixed_offsets_on_test = load_fixed_offsets_on_test
        self.vis_load_backend = vis_load_backend
        self.size_ratio = size_ratio

        vid_folder = Path(vids_dir) / 'pretrain'

        split_clip_ids_path = os.path.join(splits_path, f'lrs3_{split}.txt')
        if not os.path.exists(split_clip_ids_path):
            clip_paths = sorted(vid_folder.rglob('*/*.mp4'))
            # filter "bad" examples
            clip_paths = self.filter_bad_examples(clip_paths)
            self.make_split_files(clip_paths)

        # read the ids from a split
        split_clip_ids = sorted(open(split_clip_ids_path).read().splitlines())

        # make paths from the ids
        clip_paths = [os.path.join(vids_dir, v + '.mp4') for v in split_clip_ids]

        # loading the fixed offsets. COMMENT THIS IF YOU DON'T HAVE A FILE YET
        if load_fixed_offsets_on_test and split in ['valid', 'test']:
            logger.info(f'Using fixed offset for {split}')
            self.vid2offset_params = get_fixed_offsets(transforms, split, splits_path, 'lrs3')

        self.dataset = clip_paths
        if size_ratio is not None and 0.0 < size_ratio < 1.0:
            cut_off = int(len(self.dataset) * size_ratio)
            self.dataset = self.dataset[:cut_off]

        logger.info(f'{split} has {len(self.dataset)} items')

    def __getitem__(self, index):
        path = self.dataset[index]
        rgb, audio, meta = get_video_and_audio(path, get_meta=True, max_clip_len_sec=self.max_clip_len_sec)

        # (Tv, 3, H, W) in [0, 225], (Ta, C) in [-1, 1]
        item = {'video': rgb, 'audio': audio, 'meta': meta, 'path': path, 'targets': {}, 'split': self.split}

        # loading fixed offsets so we could evaluate on the same data each time (valid and test)
        # COMMENT THIS IF YOU DON'T HAVE A FILE YET
        if self.load_fixed_offsets_on_test and self.split in ['valid', 'test']:
            unique_id = path.replace(f'{self.vids_dir}/', '').replace(self.vids_dir, '').replace('.mp4', '')
            item['targets']['offset_sec'] = self.vid2offset_params[unique_id]['offset_sec']
            item['targets']['v_start_i_sec'] = self.vid2offset_params[unique_id]['v_start_i_sec']

        if self.transforms is not None:
            item = self.transforms(item)

        return item

    def filter_bad_examples(self, paths):
        bad = set()
        base_path = Path('./data/filtered_examples_lrs')
        lists = [open(p).read().splitlines() for p in sorted(glob(str(base_path / '*.txt')))]
        for s in lists:
            bad = bad.union(s)
        logger.info(f'Number of clips before filtering: {len(paths)}')
        video_ids = [str(i).replace(self.vids_dir, '') for i in paths]
        video_ids = [str(i).replace(f'{self.vids_dir}/', '') for i in video_ids]
        paths = sorted([r for r in video_ids if r not in bad])
        logger.info(f'Number of clips after filtering: {len(paths)}')
        return paths

    def make_split_files(self, paths):
        logger.warning(f'The split files do not exist @ {self.splits_path}. Calculating the new ones.')

        # will be splitting using videos, not clips to prevent train-test intersection
        all_vids = sorted(list(set([Path(p).parent.name for p in paths])))
        random.Random(self.seed).shuffle(all_vids)

        # 0.1: splits are 8:1:1
        hold_out_ratio = 0.1
        hold_out_size = int(len(all_vids) * hold_out_ratio)
        test_vids, train_valid_vids = all_vids[:hold_out_size], all_vids[hold_out_size:]
        valid_vids, train_vids = train_valid_vids[:hold_out_size], train_valid_vids[hold_out_size:]

        # making files
        for phase, vids in zip(['train', 'valid', 'test'], [train_vids, valid_vids, test_vids]):
            with open(os.path.join(self.splits_path, f'lrs3_{phase}.txt'), 'w') as wfile:
                for path in paths:
                    vid_name = Path(path).parent.name
                    # just in the case I forgot the trailing '/' in the path
                    unique_id = path.replace(f'{self.vids_dir}/', '').replace(self.vids_dir, '') \
                                    .replace('.mp4', '')
                    if vid_name in vids:
                        wfile.write(unique_id + '\n')

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from scripts.train_utils import get_transforms
    from utils.utils import cfg_sanity_check_and_patch
    cfg = OmegaConf.load('./configs/sparse_sync.yaml')
    cfg.data.vids_path = '/scratch/project_2000936/vladimir/data/lrs3/h264_uncropped_25fps_256side_16000hz_aac/'
    cfg.data.dataset.params.load_fixed_offsets_on_test = True

    cfg_sanity_check_and_patch(cfg)
    transforms = get_transforms(cfg)

    datasets = {
        'train': LRS3('train', cfg.data.vids_path, transforms['train'], load_fixed_offsets_on_test=False),
        'valid': LRS3('valid', cfg.data.vids_path, transforms['test'], load_fixed_offsets_on_test=False),
        'test': LRS3('test', cfg.data.vids_path, transforms['test'], load_fixed_offsets_on_test=False),
    }
    for phase in ['train', 'valid', 'test']:
        print(phase, len(datasets[phase]))

    print(datasets['train'][1]['audio'].shape, datasets['train']
          [1]['video'].shape, datasets['train'][1]['meta'])
    print(datasets['valid'][1]['audio'].shape, datasets['valid']
          [1]['video'].shape, datasets['valid'][1]['meta'])
