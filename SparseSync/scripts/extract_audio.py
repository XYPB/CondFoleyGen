import subprocess
import sys
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, '.')
from utils.utils import (apply_fn_for_loop, apply_fn_in_parallel, which_ffmpeg)


def extract_audio(path, acodec, a_fps):
    assert which_ffmpeg() != '', 'activate an env with ffmpeg/ffprobe'
    path = str(path)
    new_path = str(path).replace('.mp4', '.wav')
    # reencode the original mp4: rescale, resample video and resample audio
    cmd = f'{which_ffmpeg()}'
    cmd += ' -hide_banner -loglevel panic'  # no info/error printing
    cmd += f' -i {path}'
    cmd += f' -acodec {acodec} -ar {a_fps} -ac 1'
    cmd += f' {new_path}'
    if Path(new_path).exists():
        print('Already exists:', new_path)
    else:
        subprocess.call(cmd.split())


if __name__ == '__main__':
    cfg_cli = OmegaConf.from_cli()

    assert 'vid_dir' in cfg_cli, 'Specify path to *.mp4 files'
    vid_dir = cfg_cli.vid_dir

    # syntax: cfg.get(param, default_if_param_doesnt_exist)
    a_fps = cfg_cli.get('a_fps', 16000)
    acodec = cfg_cli.get('acodec', 'pcm_s16le')
    num_workers = cfg_cli.get('num_workers', 4)

    paths = sorted([p for p in Path(vid_dir).glob('*.mp4')])

    # apply_fn_for_loop(extract_audio, paths, acodec, a_fps)  # for debugging

    def _aud_fn(p):
        return extract_audio(p, acodec, a_fps)
    apply_fn_in_parallel(_aud_fn, paths, num_workers)
