'''assuming:
1. the videos are assumed to be from `pretrain` subset (others should also work, but test has other vids)'''
import os
import pickle
import random
import sys
import tempfile
import traceback
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import cv2
import ffmpeg
import numpy as np
import torch
import torchaudio
import torchvision
from einops import rearrange
from torchvision.transforms.functional import resize
from tqdm import tqdm

sys.path.insert(0, '.')

from dataset.dataset_utils import get_audio_stream, maybe_cache_file
from dataset.transforms import sec2frames


def get_audio_stream_extracted_from_mp4_as_wav(mp4_path, a_fps, audio_ext, a_channels, a_codec, audio_read_fn):

    def _maybe_extract_and_load_audio(cache_root, mp4_path, a_fps, audio_ext, a_channels, a_codec):
        '''if audio has not been extracted already, it will be done. Then, audio will be read (with meta)'''
        # wav_path = mp4_path.with_suffix(f'.{audio_ext}')
        wav_path = Path(cache_root / Path(mp4_path).relative_to('/')).with_suffix(f'.{audio_ext}')
        if not wav_path.exists():
            Path(wav_path.parent).mkdir(parents=True, exist_ok=True)
            process = (
                ffmpeg
                .input(str(mp4_path))
                .output(str(wav_path), acodec=a_codec, ar=a_fps, ac=a_channels)
                .global_args('-loglevel', 'panic').global_args('-nostats').global_args('-hide_banner')
            )
            try:
                process, _ = process.run()
            except ffmpeg._run.Error:
                print('ffmpeg._run.Error on', mp4_path)
        wav_path = str(wav_path)
        return audio_read_fn(wav_path, get_meta=True)

    if 'LOCAL_SCRATCH' in os.environ:
        cache_root = Path(os.environ.get('LOCAL_SCRATCH'))
        return _maybe_extract_and_load_audio(cache_root, mp4_path, a_fps, audio_ext, a_channels, a_codec)
    else:
        with tempfile.TemporaryDirectory(prefix='tempfile_') as cache_root:
            return _maybe_extract_and_load_audio(cache_root, mp4_path, a_fps, audio_ext, a_channels, a_codec)

def main():
    # lrs_meta_root = Path('/home/hdd/data/lrs3/lrs3_v0.4/')
    # full_vids_root = Path('/home/vladimir/sync/orig_full/data/lrs3_ref/video/')
    # save_root = Path('/home/vladimir/sync/orig_full/cropped_sliced/') / split
    parser = ArgumentParser()
    parser.add_argument('--split', default='pretrain')
    parser.add_argument('--lrs_meta_root', help='e.g: ./data/lrs3/lrs3_v0.4/')
    parser.add_argument('--full_vids_root', help='e.g: ./data/lrs3_ref/video/')
    parser.add_argument('--save_root', help='e.g: ./data/cropped_sliced')
    parser.add_argument('--do_face_crop', default=False, action='store_true')
    parser.add_argument('--rescale_to_px', type=int, help='rescale such that min(H, W) = `rescale_to_px`')
    args = parser.parse_args()
    print(args)

    split = args.split
    lrs_meta_root = Path(args.lrs_meta_root).resolve()
    full_vids_root = Path(args.full_vids_root).resolve()
    save_root = Path(args.save_root).resolve() / split

    if not save_root.exists():
        save_root.mkdir(parents=True, exist_ok=True)

    assert all([lrs_meta_root.exists(), full_vids_root.exists(), save_root.parent.exists()])

    full_vids_paths = sorted([Path(p) for p in full_vids_root.rglob('*.mp4')])
    random.shuffle(full_vids_paths)

    cache_meta_path = Path('./data/correct2wrong_lrs_ids_dict.pkl')
    if cache_meta_path.exists():
        correct2wrong = pickle.load(open(cache_meta_path, 'rb'))
    else:
        correct2wrong = make_vidid_correction_dict(lrs_meta_root, split)
        pickle.dump(correct2wrong, open(cache_meta_path, 'wb'))

    for vid_path in tqdm(full_vids_paths, miniters=50):
        vid_path = maybe_cache_file(vid_path)
        correct_vidid = vid_path.stem[:11]
        # the name of the meta folder sometimes doesn't correspond to
        # the corrent youtube id (_ and - are replaced with S)
        maybe_wrong_vidid = correct2wrong[correct_vidid]
        meta_paths = list((lrs_meta_root / split / maybe_wrong_vidid).rglob('*.txt'))
        random.shuffle(meta_paths)  # minimizing the chance of a collision when > 1 worker operates on one vid

        # a video usually contains several segments
        for meta_path in meta_paths:
            vidid_from_meta, frame2dim2ratio = extract_crop_info(meta_path)
            segment_id = Path(meta_path).stem
            assert vidid_from_meta == correct_vidid
            save_path_mp4 = save_root / vid_path.stem / f'{segment_id}.mp4'  # h264 rgb and 'aac' audio
            if save_path_mp4.exists():
                print(f'\nPath: {save_path_mp4} already exists. Skipping...')
            else:
                print(f'\nPath: {save_path_mp4} is missing. Processing...')
                try:
                    slice_and_crop(str(vid_path), frame2dim2ratio, save_path_mp4, args.do_face_crop, args.rescale_to_px)
                except:
                    print(f'\nError occurred when processing {save_path_mp4}:\n: {traceback.print_exc()}\n')


def slice_and_crop(vid_path: str, frame2dim2ratio: dict, save_path_mp4: Path, do_face_crop: bool,
                   rescale_to_px: Union[int, None]):
    v_fps = 25
    v_codec = 'libx264'
    # a_fps = 22050
    a_fps = 16000
    # side_size = 224
    # side_size = 256
    side_size = rescale_to_px
    audio_ext = 'wav'
    a_channels = 1
    a_codec = 'pcm_s16le'

    ## RGB (reading with cv2 and applying transforms frame-wise)
    video_reader = cv2.VideoCapture(vid_path)
    assert video_reader.get(cv2.CAP_PROP_FPS) == v_fps
    rgb_crops = []
    new_crop_meta = {}
    frame_id = 0
    status = True
    while status:
        status, frame = video_reader.read()
        if status:
            # saving frames from the segment
            if frame_id in frame2dim2ratio:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if do_face_crop:
                    cropped_frame, rgb_crop_coords = crop_image(frame, **frame2dim2ratio[frame_id])
                else:
                    cropped_frame, rgb_crop_coords = frame, dict(X=0., Y=0., H=1., W=1.)
                rgb_crops.append(cropped_frame)
                new_crop_meta[frame_id] = rgb_crop_coords
            frame_id += 1
    video_reader.release()
    assert len(rgb_crops) > 0, vid_path
    # converts to torch tensor and resizes
    if side_size is None:
        rgb_segment_crop = rearrange(torch.from_numpy(np.array(rgb_crops)), 't h w c -> t c h w')
    else:
        rgb_segment_crop = [resize(rearrange(torch.from_numpy(x), 'h w c -> c h w'), side_size) for x in rgb_crops]

    # rgb_segment_crop = torch.randint_like(rgb_segment_crop, 0, 256)
    # print(rgb_segment_crop.shape)
    # rgb_segment_crop = torch.randint(0, 256, (100, 1, 1, 1)).tile((1, 3, 224, 224)).repeat_interleave(repeats=5, dim=0)[:rgb_segment_crop.shape[0]]

    ## AUDIO
    audio, meta = get_audio_stream_extracted_from_mp4_as_wav(vid_path, a_fps, audio_ext, a_channels,
                                                             a_codec, get_audio_stream)
    # trims audio using rgb frame positions
    audio_segment, aud_segment_coords = segment_audio(audio, frame2dim2ratio, v_fps, a_fps)
    # audio_segment = torch.rand_like(audio_segment) * 2 - 1
    # print(audio_segment.shape)
    # audio_segment = torch.rand(16000).repeat_interleave(repeats=512)[:audio_segment.shape[0]] * 2 - 1

    ## Save files
    save_path_txt = save_path_mp4.with_suffix('.txt')  # for meta
    save_path_wav = save_path_mp4.with_suffix('.wav')  # waveform with the `a_codec` format
    save_path_mp4.parent.mkdir(exist_ok=True, parents=True)

    if save_path_mp4.exists():
        print(f'\nPath: {save_path_mp4} has became `already exist`. Skipping...')
    else:
        # mp4 and separately audio
        rgb_segment_crop = rearrange(rgb_segment_crop, 'n c h w -> n h w c')
        audio_segment = audio_segment[None, :]  # adding the channels dim
        # avoiding (libx264) "height not divisible by 2"
        adj_h, adj_w = rgb_segment_crop.shape[1]//2*2, rgb_segment_crop.shape[2]//2*2
        rgb_segment_crop = rgb_segment_crop[:, :adj_h, :adj_w, :]
        torchvision.io.write_video(str(save_path_mp4), rgb_segment_crop, v_fps, v_codec,
                                   audio_array=audio_segment, audio_fps=a_fps, audio_codec='aac')
        # saving the segment in a raw format as specified by `a_codec` because write_video will save it as aac
        torchaudio.save(save_path_wav, audio_segment, a_fps, format='wav')

        ## new meta (crop info for audio and the new info for RGB)
        with open(save_path_txt, 'w') as fwrite:
            fwrite.write(f'Ref: {Path(vid_path).stem}\n')
            fwrite.write('AUDIO SEGMENT (start_sec duration_sec):\n')
            fwrite.write(f'{aud_segment_coords["start_sec"]:.2f} {aud_segment_coords["duration_sec"]:.2f}\n')
            fwrite.write('RGB:\n')
            fwrite.write('FRAME X Y W H\n')
            for FRAME in sorted(list(new_crop_meta.keys())):
                crp_crd = new_crop_meta[FRAME]
                X, Y, W, H = crp_crd['X'], crp_crd['Y'], crp_crd['W'], crp_crd['H']
                fwrite.write(f'{FRAME:06d} {X:.3f} {Y:.3f} {W:.3f} {H:.3f}\n')

def segment_audio(audio: torch.Tensor, frame2dim2ratio: dict, v_fps: int, a_fps: int) -> tuple:
    start_sec = min(frame2dim2ratio.keys()) / v_fps
    end_sec = max(frame2dim2ratio.keys()) / v_fps
    assert end_sec > start_sec, (start_sec, end_sec)
    start_a_frames = sec2frames(start_sec, a_fps)
    end_a_frames = sec2frames(end_sec, a_fps)
    audio_segment = audio[start_a_frames:end_a_frames]
    crop_coords = dict(start_sec=start_sec, duration_sec=end_sec-start_sec)
    return audio_segment, crop_coords


def crop_image(img, x: float, y: float, h: float, w: float) -> tuple:
    '''In addition to the crop provided by LRS3, it pads the bbox to a square (adapts for boundaries)'''
    img_h, img_w, C = img.shape

    # ratio -> pixels
    crop_y, crop_x = int(img_h * y), int(img_w * x)
    crop_h, crop_w = int(img_h * h), int(img_w * w)

    # padding to the largest side
    crop_cx, crop_cy = crop_x + crop_w / 2, crop_y + crop_h / 2
    max_side = max(crop_h, crop_w)
    crop_new_x, crop_new_y = crop_cx - max_side / 2, crop_cy - max_side / 2
    # making sure x and y are within the image and if crop is applied it will be still withing the image
    crop_new_x, crop_new_y = clamp(crop_new_x, 0, img_w-max_side), clamp(crop_new_y, 0, img_h-max_side)
    assert (0 < crop_h <= img_h) and (0 < crop_w < img_w), (crop_h, crop_w, img_h, img_w)
    assert (0 <= crop_new_x <= img_w) and (0 <= crop_new_y <= img_h), (crop_new_x, crop_new_y, img_w, img_h)
    crop_y, crop_x = int(crop_new_y), int(crop_new_x)
    crop_h, crop_w = max_side, max_side

    scaled_crop_coords = dict(X=crop_x/img_w, Y=crop_y/img_h, H=max_side/img_h, W=max_side/img_w)

    return img[crop_y:(crop_y + crop_h), crop_x:(crop_x + crop_w), :], scaled_crop_coords

def clamp(n, smallest, largest):
    '''makes sure `n` is in the [`smallest`, `largest`] interval'''
    return max(smallest, min(n, largest))

def make_vidid_correction_dict(lrs_meta_root: Path, split: str):
    ''' # the name of the folder does not correspond to the youtube video is (_ and - are replaced with S) '''
    meta_files_paths = sorted([Path(p) for p in (lrs_meta_root / split).rglob('*/*.txt')])
    correct2wrong = {extract_crop_info(p)[0]: p.parent.stem for p in tqdm(meta_files_paths)}
    return correct2wrong


def extract_crop_info(path):
    frame2dim2ratio = {}
    with open(path) as rfile:
        saw_crops = False
        for line in rfile.readlines():
            line = line.strip()
            if line.startswith(('Text:', 'Conf:')):
                continue
            if line.startswith(('Ref:', )):
                _, correct_vid_id = line.split()
                continue
            if line.startswith('FRAME'):
                saw_crops = True
                continue
            if len(line) == 0:
                if saw_crops:
                    break
                else:
                    continue
            frame, x, y, w, h = line.split()
            x, y, w, h = map(float, [x, y, w, h])
            # sometimes the bbox is out video boundaries (crop w or h > 1): ~0.1% of such segments
            w, h = min(1., w), min(1., h)
            # sometimes the bbox starts out of boundaries (x, y < 0): ~2.5% of such segments
            x, y = max(0., x), max(0., y)
            frame2dim2ratio[int(frame)] = dict(x=x, y=y, w=w, h=h)
    # the name of the folder does not correspond to the youtube video is (_ and - are replaced)
    return correct_vid_id, frame2dim2ratio


if __name__ == '__main__':
    main()
