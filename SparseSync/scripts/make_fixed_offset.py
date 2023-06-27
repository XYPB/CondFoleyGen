''' run as: `python ./scripts/make_fixed_offset.py config=./configs/av_sync.yaml` '''
from pathlib import Path
import sys
import os

import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm
import csv

sys.path.insert(0, '.')  # nopep8
import random

from torch.utils.data import DataLoader
from utils.utils import cfg_sanity_check_and_patch, instantiate_from_config, get_fixed_off_fname
from scripts.train_utils import get_datasets


def main(cfg: dict):
    # shifting the state to avoid similar offsets for the valid (or test) as during the 1st epoch of training
    random.seed(cfg.training.seed + 1)

    # filtering the sequence of transforms used for training.
    data_transforms = []
    transform_sequence_cfg = cfg.get('transform_sequence_train', None)
    for t_cfg in transform_sequence_cfg:
        # This is very fragile for changes in transforms (order, additions)
        # if t_cfg['target'].endswith(('.EqualifyFromRight', '.RGBSpatialCrop', '.TemporalCropAndOffset')):
        # if t_cfg['target'].endswith(('.EqualifyFromRight', '.RGBSpatialCrop', '.TemporalCropAndOffsetBalanced')):
        # if t_cfg['target'].endswith(('.EqualifyFromRight', '.RGBSpatialCrop', '.TemporalCropAndOffsetRandomFeasible')):
        if t_cfg['target'].endswith(('.EqualifyFromRight', '.RGBSpatialCropSometimesUpscale', '.TemporalCropAndOffsetRandomFeasible')):
            data_transforms.append(instantiate_from_config(t_cfg))
    assert len(data_transforms) == 3, 'has anything changed in the trasforms sequence?'
    data_transforms = torchvision.transforms.Compose(data_transforms)

    datasets = get_datasets(cfg, {k: data_transforms for k in ['train', 'valid', 'test']})

    # num_workers is 0 to make sure the results are reproducible with the seed
    loaders = {
        'train': DataLoader(datasets['train'], batch_size=8, num_workers=8),
        'valid': DataLoader(datasets['valid'], batch_size=8, num_workers=8),
        'test': DataLoader(datasets['test'], batch_size=8, num_workers=8),
    }

    for phase in ['valid', 'test', 'train']:
        offs_from_each_batch = []
        vstarts_from_each_batch = []
        paths = []

        prog_bar = tqdm(loaders[phase], phase, ncols=0)
        for i, batch in enumerate(prog_bar):
            offset_sec = batch['targets']['offset_sec']
            v_start_i_sec = batch['targets']['v_start_i_sec']
            offs_from_each_batch += [offset_sec.detach().half().cpu()]
            vstarts_from_each_batch += [v_start_i_sec.half().cpu()]
            paths += batch['path']

        offs = torch.cat(offs_from_each_batch)
        vstarts = torch.cat(vstarts_from_each_batch)

        fixed_offset_fname = get_fixed_off_fname(data_transforms, phase)

        vids_path = cfg.data.vids_path

        if '/vggsound/' in vids_path:
            suffix = '_vggsound'

            def _get_unique_id(path):
                return Path(path).stem
        elif '/lrs3/' in vids_path:
            suffix = '_lrs3'

            def _get_unique_id(path):
                return path.replace(f'{vids_path}/', '').replace(vids_path, '').replace('.mp4', '')

        fixed_offset_path = os.path.join(f'./data/fixed_offsets{suffix}', fixed_offset_fname)
        with open(fixed_offset_path, 'w') as outf:
            writer = csv.writer(outf)
            writer.writerow(['path', 'offset_sec', 'vstart_sec'])
            for p, o, s in zip(paths, offs, vstarts):
                writer.writerow([_get_unique_id(p), round(o.item(), 2), round(s.item(), 2)])


if __name__ == '__main__':
    ''' run as: (COMMENT VGGSOUND) `python ./scripts/make_fixed_offset.py config=./configs/sparse_sync.yaml` '''
    '''
        You can use this script to create fixed offsets in `./data/` but before that you need to
        comment (turn off) parts that load fixed offsets in `./dataset/lrs` and `./dataset/vggsound`.
        Run it as:

        python ./scripts/make_fixed_offset.py \
            config=./configs/sparse_sync.yaml \
            off_cls_num="21" \
            crop_len_sec="5" \
            max_off_sec="2" \
            dataset="dataset.vggsound.VGGSound" \
            vids_path="/path/to/h264_video_25fps_256side_16000hz_aac"
    '''
    cfg_cli = OmegaConf.from_cli()
    cfg_yml = OmegaConf.load(cfg_cli.config)
    # the latter arguments are prioritized
    cfg = OmegaConf.merge(cfg_yml, cfg_cli)

    # cfg.data.vids_path = '/home/nvme/data/vggsound/video_10fps_256side_22050hz'
    # cfg.data.dataset.target = 'dataset.vggsound.VGGSound'
    # cfg.data.vids_path = '/scratch/project_2000936/vladimir/vggsound/h264_video_25fps_256side_16000hz_aac'
    # cfg.data.vids_path = '/home/nvme/data/lrs3/video_10fps_256side_22050hz'
    cfg.data.vids_path = cfg_cli.vids_path
    cfg.data.dataset.target = cfg_cli.dataset
    cfg.data.crop_len_sec = cfg_cli.crop_len_sec
    cfg.data.max_off_sec = cfg_cli.max_off_sec
    # for cls_num in [3, 9, 41]:  # [-2, 0, +2]
    # for cls_num in [3, 5, 21]:   # [-1, 0, +1]
    cfg.model.params.transformer.params.num_offset_cls = cfg_cli.off_cls_num
    cfg.data.audio_jitter_sec = 0.0  # turn off augmentations

    cfg_sanity_check_and_patch(cfg)
    print(OmegaConf.to_yaml(cfg))

    main(cfg)
