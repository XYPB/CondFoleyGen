import csv
import os
from pathlib import Path
from time import sleep
import torchaudio
import torchvision
from glob import glob
import shutil
from utils.utils import get_fixed_off_fname


def get_fixed_offsets(transforms, split, splits_path, dataset_name):
    '''dataset_name: `vggsound` or `lrs3`'''
    vid2offset_params = {}
    fixed_offset_fname = get_fixed_off_fname(transforms, split)
    fixed_offset_path = os.path.join(splits_path, f'fixed_offsets_{dataset_name}', fixed_offset_fname)
    fixed_offset_paths = sorted(glob(fixed_offset_path.replace(split, '*')))
    assert len(fixed_offset_paths) > 0, f'Perhaps: {fixed_offset_path} does not exist. Make fixed offsets'

    for fix_off_path in fixed_offset_paths:
        reader = csv.reader(open(fix_off_path))
        # skipping the header
        next(reader)
        for v, o, s in reader:
            assert v not in vid2offset_params, 'otherwise, offsets from other splits will override each other'
            vid2offset_params[v] = {'offset_sec': float(o), 'v_start_i_sec': float(s)}
    return vid2offset_params


def maybe_cache_file(path: os.PathLike):
    '''Motivation: if every job reads from a shared disk it`ll get very slow, consider an image can
    be 2MB, then with batch size 32, 16 workers in dataloader you`re already requesting 1GB!! -
    imagine this for all users and all jobs simultaneously.'''
    # checking if we are on cluster, not on a local machine
    if 'LOCAL_SCRATCH' in os.environ:
        cache_dir = os.environ.get('LOCAL_SCRATCH')
        # a bit ugly but we need not just fname to be appended to `cache_dir` but parent folders,
        # otherwise the same fnames in multiple folders will create a bug (the same input for multiple paths)
        cache_path = os.path.join(cache_dir, Path(path).relative_to('/'))
        if not os.path.exists(cache_path):
            os.makedirs(Path(cache_path).parent, exist_ok=True)
            shutil.copyfile(path, cache_path)
        return cache_path
    else:
        return path


def get_video_and_audio(path, get_meta=False, max_clip_len_sec=None):
    path = maybe_cache_file(path)
    # try-except was meant to solve issue when `maybe_cache_file` copies a file but another worker tries to
    # load it because it thinks that the file exists. However, I am not sure if it works :/.
    # Feel free to refactor it.
    try:
        rgb, audio, meta = torchvision.io.read_video(path, pts_unit='sec', end_pts=max_clip_len_sec)
        meta['video_fps']
    except KeyError:
        print(f'Problem at {path}. Trying to wait and load again...')
        sleep(5)
        rgb, audio, meta = torchvision.io.read_video(path, pts_unit='sec', end_pts=max_clip_len_sec)
        meta['video_fps']
    # (T, 3, H, W) [0, 255, uint8] <- (T, H, W, 3)
    rgb = rgb.permute(0, 3, 1, 2)
    # (Ta) <- (Ca, Ta)
    audio = audio.mean(dim=0)
    # FIXME: this is legacy format of `meta` as it used to be loaded by VideoReader.
    try:
        meta['audio_fps']
    except KeyError:
        meta['audio_fps'] = 16000
    meta = {
        'video': {'fps': [meta['video_fps']]},
        'audio': {'framerate': [meta['audio_fps']]},
    }
    return rgb, audio, meta


def get_audio_stream(path, get_meta=False):
    '''Used only in feature extractor training'''
    path = str(Path(path).with_suffix('.wav'))
    path = maybe_cache_file(path)
    waveform, _ = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if get_meta:
        info = torchaudio.info(path)
        duration = info.num_frames / info.sample_rate
        meta = {'audio': {'duration': [duration], 'framerate': [info.sample_rate]}}
        return waveform, meta
    else:
        return waveform
