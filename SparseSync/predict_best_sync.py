import subprocess
import os
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
import torchaudio
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm

from librosa.effects import time_stretch
from dataset.dataset_utils import get_video_and_audio
from dataset.transforms import make_class_grid, quantize_offset
from model.modules.attn_recorder import Recorder
from model.modules.feature_selector import CrossAttention
from utils.utils import which_ffmpeg, check_if_file_exists_else_download
from scripts.example import reencode_video, reconstruct_video_from_input, decode_single_video_prediction, decode_single_video_best_prediction_clean
from scripts.train_utils import get_model, get_transforms, prepare_inputs


def attach_audio_to_video(origin_path, wav, dest, SR):
    # clip = ImageSequenceClip([f for f in frames], fps=FPS)
    wav_dest = os.path.join('tmp', dest.split('/')[-1].replace('.mp4', '.wav'))
    sf.write(wav_dest, wav, SR)
    # clip.write_videofile('tmp.mp4', audio=False, fps=FPS, verbose=False, logger=None, audio_fps=SR)
    cmd = f'{which_ffmpeg()} -i {origin_path} -i {wav_dest} -hide_banner -loglevel panic -map 0:v -map 1:a -c:v copy -shortest -y {dest}'
    subprocess.call(cmd.split())
    return

def example(exp_name, vid_path, vfps, afps, device, input_size, v_start_i_sec, offset_sec):
    cfg_path = f'./logs/sync_models/{exp_name}/cfg-{exp_name}.yaml'
    ckpt_path = f'./logs/sync_models/{exp_name}/{exp_name}.pt'
    
    # if the model does not exist try to download it from the server
    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)
    
    # load config
    cfg = OmegaConf.load(cfg_path)
    
    # checking if the provided video has the correct frame rates
    print(f'Using video: {vid_path}')
    v, a, vid_meta = torchvision.io.read_video(vid_path, pts_unit='sec')
    T, H, W, C = v.shape
    if vid_meta['video_fps'] != vfps or vid_meta['audio_fps'] != afps or min(H, W) != input_size:
        print(f'Reencoding. vfps: {vid_meta["video_fps"]} -> {vfps};', end=' ')
        print(f'afps: {vid_meta["audio_fps"]} -> {afps};', end=' ')
        print(f'{(H, W)} -> min(H, W)={input_size}')
        vid_path = reencode_video(vid_path, vfps, afps, input_size)
    else:
        print(f'No need to reencode. vfps: {vid_meta["video_fps"]}; afps: {vid_meta["audio_fps"]}; min(H, W)={input_size}')
    
    device = torch.device(device)
    
    # load the model
    _, model = get_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])
    
    # Recorder wraps the model to access attention values
    # type2submodule = {'rgb': 'v_selector', 'audio': 'a_selector'}
    submodule_name = 'v_selector'  # else 'v_selector'
    model = Recorder(model, watch_module=CrossAttention, submodule_name=submodule_name)
    
    model.eval()
    
    # load visual and audio streams
    # (Tv, 3, H, W) in [0, 225], (Ta, C) in [-1, 1]
    rgb, audio, meta = get_video_and_audio(vid_path, get_meta=True)
    print(audio.shape)
    
    # TODO: check if the offset and start are zeros and print this
    # making an item (dict) to apply transformations
    item = {
        'video': rgb, 'audio': audio, 'meta': meta, 'path': vid_path, 'split': 'test',
        'targets': {
            # setting the start of the visual crop and the offset size.
            # For instance, if the model is trained on 5sec clips, the provided video is 9sec, and `v_start_i_sec=1.3`
            # the transform will crop out a 5sec-clip from 1.3 to 6.3 seconds and shift the start of the audio
            # track by `offset_sec` seconds. It means that if `offset_sec` > 0, the audio will
            # start `offset_sec` earlier than the rgb track.
            # It is a good idea to use something in [-`max_off_sec`, `max_off_sec`] (see `grid`)
            'v_start_i_sec': v_start_i_sec,
            'offset_sec': offset_sec,
            # dummy values -- don't mind them
            'vggsound_target': 0,
            'vggsound_label': 'PLACEHOLDER',
        },
    }
    
    # making the offset class grid similar to the one used in transforms
    max_off_sec = cfg.data.max_off_sec
    grid = make_class_grid(-max_off_sec, max_off_sec, cfg.model.params.transformer.params.num_offset_cls)
    # TODO: maybe?
    # assert min(grid) <= offset_sec <= max(grid)
    
    # applying the transform
    transforms = get_transforms(cfg)['test']
    item = transforms(item)
    
    # prepare inputs for inference
    batch = torch.utils.data.default_collate([item])
    aud, vid, targets = prepare_inputs(batch, device)
    
    # sanity check: we will take the input to the `model` and recontruct make a video from it.
    # Use this check to make sure the input makes sense (audio should be ok but shifted as you specified)
    reconstruct_video_from_input(aud, vid, batch['meta'], vid_path, v_start_i_sec, offset_sec, vfps, afps)
    
    # forward pass
    _, off_logits, attention = model(vid, aud, targets)
    
    # simply prints the results of the prediction
    probs = decode_single_video_prediction(off_logits, grid, item)
    print(probs)


def predict_best_sync(exp_name, multiple_audio_path, vfps, afps, device, input_size, args):
    cfg_path = f'./logs/sync_models/{exp_name}/cfg-{exp_name}.yaml'
    ckpt_path = f'./logs/sync_models/{exp_name}/{exp_name}.pt'
    
    # if the model does not exist try to download it from the server
    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)
    
    # load config
    cfg = OmegaConf.load(cfg_path)
    
    device = torch.device(device)
    
    # load the model
    _, model = get_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])
    
    # Recorder wraps the model to access attention values
    # type2submodule = {'rgb': 'v_selector', 'audio': 'a_selector'}
    submodule_name = 'v_selector'  # else 'v_selector'
    model = Recorder(model, watch_module=CrossAttention, submodule_name=submodule_name)
    
    model.eval()
    
    dest_dir = args.dest_dir
    if dest_dir == None:
        dest_dir = os.path.dirname(multiple_audio_path)
    os.makedirs(dest_dir, exist_ok=True)
    if args.scale != 1:
        folder = f'full_generated_video_scale{args.scale:.1f}'
    else:
        folder = 'full_generated_video'
    os.makedirs(os.path.join(dest_dir, folder), exist_ok=True)

    generated_audios = torch.load(multiple_audio_path)
    pbar = tqdm(generated_audios.items(), total=len(generated_audios))
    for origin_path, cond_group in pbar:
        # checking if the provided video has the correct frame rates
        v, a, vid_meta = torchvision.io.read_video(origin_path, pts_unit='sec')
        T, H, W, C = v.shape
        if vid_meta['video_fps'] != vfps or vid_meta['audio_fps'] != afps or min(H, W) != input_size:
            if args.time_stretch:
                origin_path_reencode = reencode_video(origin_path, 62.5, afps, input_size)
            else:
                origin_path_reencode = reencode_video(origin_path, vfps, afps, input_size)
        else:
            pass

        # load visual and audio streams
        # (Tv, 3, H, W) in [0, 225], (Ta, C) in [-1, 1]
        rgb, audio, meta = get_video_and_audio(origin_path_reencode, get_meta=True)
        meta['video']['fps'] = [25,]
        resampler = torchaudio.transforms.Resample(22050, afps)

        for cond_path, audios in cond_group.items():
            best_sync_idx = -1
            min_shift = torch.inf
            cur_prob = -1
            for idx, wav in enumerate(audios):
                # to_record = False
                # scale
                wav = args.scale * wav
                wav = np.clip(wav, -1.0, 1.0)
                if args.time_stretch:
                    wav = time_stretch(wav, rate=0.4)
                wav = torch.FloatTensor(wav).squeeze(0)
                wav = resampler(wav)
                item = {
                    'video': rgb, 'audio': wav, 'meta': meta, 'path': origin_path_reencode, 'split': 'test',
                    'targets': {
                        'v_start_i_sec': 0.0,
                        'offset_sec': 0.0,
                        # dummy values -- don't mind them
                        'vggsound_target': 0,
                        'vggsound_label': 'PLACEHOLDER',
                    },
                }
                
                # making the offset class grid similar to the one used in transforms
                max_off_sec = cfg.data.max_off_sec
                grid = make_class_grid(-max_off_sec, max_off_sec, cfg.model.params.transformer.params.num_offset_cls)
                
                # applying the transform
                transforms = get_transforms(cfg)['test']
                item = transforms(item)
                
                # prepare inputs for inference
                batch = torch.utils.data.default_collate([item])
                aud, vid, targets = prepare_inputs(batch, device)
                
                # sanity check: we will take the input to the `model` and recontruct make a video from it.
                # Use this check to make sure the input makes sense (audio should be ok but shifted as you specified)
                # reconstruct_video_from_input(aud, vid, batch['meta'], origin_path_reencode, v_start_i_sec, offset_sec, vfps, afps)
                # forward pass
                with torch.no_grad():
                    _, off_logits, attention = model(vid, aud, targets)
                # simply prints the results of the prediction
                top_prob, top_shift = decode_single_video_best_prediction_clean(off_logits, grid, item)
                if exp_name == '22-09-21T21-00-52':
                    if abs(top_shift) <= args.tolerance and top_prob > cur_prob:
                        best_sync_idx = idx
                        cur_prob = top_prob
                        min_shift = top_shift
                        # to_record = True
                elif exp_name == '22-08-18T09-44-31':
                    if abs(top_shift) < abs(min_shift):
                        best_sync_idx = idx
                        cur_prob = top_prob
                        min_shift = top_shift
                    elif abs(top_shift) == abs(min_shift) and top_prob > cur_prob:
                        best_sync_idx = idx
                        cur_prob = top_prob
                        min_shift = top_shift
                pbar.set_description_str(f'idx: {idx}, cur prob: {top_prob:.2f}, cur shift: {top_shift:.2f}, best prob: {cur_prob:.2f} min shift: {min_shift:2f}')
            attach_audio_to_video(origin_path, audios[best_sync_idx], os.path.join(dest_dir, folder, Path(origin_path).stem + '_to_' + Path(cond_path).stem + '.mp4'), SR=22050)

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tolerance', type=float, default=1e-5)
parser.add_argument('--time_stretch', action='store_true')
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--cnt', type=int, default=25)
parser.add_argument('--scale', type=float, default=1)
parser.add_argument('--dest_dir', type=str, default='logs/re-ranking_generation_result')


if __name__ == '__main__':
    args = parser.parse_args()
    # exp_name = '22-08-18T09-44-31'
    exp_name = '22-09-21T21-00-52'
    # vid_path = './data/vggsound/h264_video_25fps_256side_16000hz_aac/3qesirWAGt4_20000_30000.mp4'  # dog barking
    device = f'cuda:{args.device}'
    # target values for an input video (the video will be reencoded to match these)
    vfps = 25
    afps = 16000
    input_size = 256
    # you may artificially offset the audio and visual tracks:
    v_start_i_sec = 0.0  # start of the visual track
    offset_sec = 0.0  # how early audio should start than the visual track

    # example(exp_name, vid_path, vfps, afps, device, input_size, v_start_i_sec, offset_sec)
    predict_best_sync(exp_name, os.path.join(args.dest_dir, f'{args.cnt}_times_split_{args.split}_wav_dict.pt'), vfps, afps, device, input_size, args)