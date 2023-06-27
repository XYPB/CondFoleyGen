import difflib
import importlib
import subprocess
from multiprocessing import Pool
from pathlib import Path

import requests
from omegaconf import OmegaConf
from tqdm import tqdm

PARENT_LINK = 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
FNAME2LINK = {
    # feature extractors
    'ResNetAudio-22-08-04T09-51-04.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-08-04T09-51-04.pt',  # 2s
    'ResNetAudio-22-08-03T23-14-49.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-08-03T23-14-49.pt',  # 3s
    'ResNetAudio-22-08-03T23-14-28.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-08-03T23-14-28.pt',  # 4s
    'ResNetAudio-22-06-24T08-10-33.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-06-24T08-10-33.pt',  # 5s
    'ResNetAudio-22-06-24T17-31-07.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-06-24T17-31-07.pt',  # 6s
    'ResNetAudio-22-06-24T23-57-11.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-06-24T23-57-11.pt',  # 7s
    'ResNetAudio-22-06-25T04-35-42.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-06-25T04-35-42.pt',  # 8s
    # ft VGGSound-Full
    '22-09-21T21-00-52.pt': f'{PARENT_LINK}/sync/sync_models/22-09-21T21-00-52/22-09-21T21-00-52.pt',
    'cfg-22-09-21T21-00-52.yaml': f'{PARENT_LINK}/sync/sync_models/22-09-21T21-00-52/cfg-22-09-21T21-00-52.yaml',
    # ft VGGSound-Sparse
    '22-07-28T15-49-45.pt': f'{PARENT_LINK}/sync/sync_models/22-07-28T15-49-45/22-07-28T15-49-45.pt',
    'cfg-22-07-28T15-49-45.yaml': f'{PARENT_LINK}/sync/sync_models/22-07-28T15-49-45/cfg-22-07-28T15-49-45.yaml',
    # only pt on LRS3
    '22-07-13T22-25-49.pt': f'{PARENT_LINK}/sync/sync_models/22-07-13T22-25-49/22-07-13T22-25-49.pt',
    'cfg-22-07-13T22-25-49.yaml': f'{PARENT_LINK}/sync/sync_models/22-07-13T22-25-49/cfg-22-07-13T22-25-49.yaml',
}

def check_if_file_exists_else_download(path, chunk_size=1024):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
        with requests.get(FNAME2LINK[path.name], stream=True) as r:
            total_size = int(r.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                with open(path, 'wb') as f:
                    for data in r.iter_content(chunk_size=chunk_size):
                        if data:
                            f.write(data)
                            pbar.update(chunk_size)


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library
    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if 'target' not in config:
        raise KeyError('Expected key `target` to instantiate.')
    return get_obj_from_str(config['target'])(**config.get('params', dict()))

def fix_prefix(prefix):
    if len(prefix) > 0:
        prefix += '_'
    return prefix

def cfg_sanity_check_and_patch(cfg):
    assert not (cfg.training.resume and cfg.training.finetune), 'it is either funetuning or resuming'
    assert not (cfg.training.resume and cfg.training.run_test_only), 'it is either resuming or testing-only'
    assert not (cfg.training.finetune and cfg.training.run_test_only), 'it is either finetune or testing-only'

    if cfg.data.dataset.params.get('iter_times', 1) > 1:
        assert cfg.data.dataset.params.load_fixed_offsets_on_test == False, 'iterating on the same data'

    if cfg.training.resume or cfg.training.run_test_only or cfg.training.finetune:
        assert Path(cfg.ckpt_path).exists(), cfg.ckpt_path
    if cfg.training.resume:
        assert Path(cfg.logging.logdir, cfg.start_time).exists(), Path(cfg.logging.logdir, cfg.start_time)
    if cfg.action in ['train_avsync_model', 'train_avsync_01']:
        vfeat_extractor_target = cfg.model.params.vfeat_extractor.target
        if vfeat_extractor_target.endswith('S3DVisualFeatures'):
            # S3D bridge 1024 -> 512 should be present
            v_bridge_cfg = cfg.model.params.v_bridge_cfg
            assert v_bridge_cfg.target.endswith(('AppendZerosToHidden', 'ConvBridgeVisual')), 'S3D bridge?'
            assert v_bridge_cfg.params.in_channels == 1024, 'S3D bridge?'


def get_fixed_off_fname(data_transforms, split):
    for t in data_transforms.transforms:
        if hasattr(t, 'class_grid'):
            min_off = t.class_grid.min().item()
            max_off = t.class_grid.max().item()
            grid_size = len(t.class_grid)
            crop_len_sec = t.crop_len_sec
            return f'{split}_size{grid_size}_crop{crop_len_sec}_min{min_off:.2f}_max{max_off:.2f}.csv'


def disable_print_if_not_master(is_master):
    """
    from: https://github.com/pytorch/vision/blob/main/references/video_classification/utils.py
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def apply_fn_for_loop(fn, lst, *args):
    for path in tqdm(lst):
        fn(path, *args)


def apply_fn_in_parallel(fn, lst, num_workers):
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(fn, lst), total=len(lst)))


def show_cfg_diffs(a, b, save_diff_path=None):
    a = OmegaConf.to_yaml(a).split('\n')
    b = OmegaConf.to_yaml(b).split('\n')

    if save_diff_path is None:
        for line in difflib.unified_diff(a, b, fromfile='old', tofile='new', lineterm=''):
            print(line)
    else:
        with open(save_diff_path, 'w') as wfile:
            for line in difflib.unified_diff(a, b, fromfile='old', tofile='new', lineterm=''):
                wfile.write(f'{line}\n')
