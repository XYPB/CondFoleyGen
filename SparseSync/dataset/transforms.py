import logging
import random
from typing import Tuple
import torch
import torchvision
import torchaudio
import numpy as np

logger = logging.getLogger(f'main.{__name__}')


def sec2frames(sec, fps):
    return int(sec * fps)

def frames2sec(frames, fps):
    return frames / fps

class MaybeTrimToVGGSoundCrop(torch.nn.Module):

    def __init__(self, vggs_crop_size_sec: int) -> None:
        super().__init__()
        self.vggs_crop_size_sec = vggs_crop_size_sec

    def forward(self, item):
        if 'rel_start_sec' in item['meta']:
            rel_start_sec = item['meta']['rel_start_sec']
            v_fps = item['meta']['video']['fps'][0]
            a_fps = item['meta']['audio']['framerate'][0]
            a_start_frames = sec2frames(rel_start_sec, a_fps)
            v_start_frames = sec2frames(rel_start_sec, v_fps)
            a_end_frames = a_start_frames + a_fps * self.vggs_crop_size_sec
            v_end_frames = v_start_frames + v_fps * self.vggs_crop_size_sec
            item['video'] = item['video'][v_start_frames:v_end_frames]
            item['audio'] = item['audio'][a_start_frames:a_end_frames]
        return item

class EqualifyFromRight(torch.nn.Module):

    def __init__(self, clip_max_len_sec=10):
        '''
        Takes the dataset item and makes sure more streams are of an equal size in terms of fps.
        It, however, assumes that the signal is synched and trims the ending parts ('from the right').
        '''
        super().__init__()
        self.clip_max_len_sec = clip_max_len_sec

    def forward(self, item):
        '''
        `item`: {'video': (Tv, C, H, W), 'audio': (Ta,),
                 'meta': {
                     'audio': {'framerate': [float], 'duration': [float]}
                     'video': {'fps': [float], 'duration': [float]}}
        '''
        a_fps = item['meta']['audio']['framerate'][0]
        v_fps = item['meta']['video']['fps'][0]

        Ta = item['audio'].shape[0]
        Tv, C, H, W = item['video'].shape

        a_len_secs = Ta / a_fps
        v_len_secs = Tv / v_fps
        min_len = min(self.clip_max_len_sec, a_len_secs, v_len_secs)

        a_frames_per_v_frame = a_fps // v_fps
        v_len_frames = int(v_fps * min_len)
        a_len_frames = int(a_frames_per_v_frame * v_len_frames)
        # print(a_len_frames, v_len_frames)

        assert a_len_frames <= Ta and v_len_frames <= Tv

        item['audio'] = item['audio'][:a_len_frames]
        item['video'] = item['video'][:v_len_frames, :, :, :]

        return item


class AudioResampleDynamic(torch.nn.Module):
    '''
    Using the functional form (slower) because not all audios are resampled equally.
    '''

    def __init__(self, new_freq):
        super().__init__()
        self.new_freq = new_freq

    def forward(self, item):
        a_fps_orig = item['meta']['audio']['framerate'][0]
        if a_fps_orig != self.new_freq:
            logger.warning(f'Video: {item["path"]} has framerate: {a_fps_orig} changing to {self.new_freq}')
            item['audio'] = torchaudio.functional.resample(item['audio'], a_fps_orig, self.new_freq)
            item['meta']['audio']['framerate'][0] = self.new_freq
        return item


class RGBSpatialCrop(torch.nn.Module):

    def __init__(self, input_size, is_random):
        super().__init__()
        assert input_size is not None, f'smaller_input_size is `{input_size}`'
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.is_random = is_random

    @staticmethod
    def get_random_crop_sides(vid, output_size):
        '''Slice parameters for random crop'''
        h, w = vid.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    @staticmethod
    def get_center_crop_sides(vid, output_size):
        '''Slice parameters for center crop'''
        h, w = vid.shape[-2:]
        th, tw = output_size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return i, j, th, tw

    def forward(self, item):
        # (Tv, C, H, W)
        vid = item['video']
        item['meta']['video']['precrop_shape'] = vid.shape
        if self.is_random:
            i, j, h, w = self.get_random_crop_sides(vid, self.input_size)
        else:
            i, j, h, w = self.get_center_crop_sides(vid, self.input_size)
        item['video'] = vid[..., i:(i + h), j:(j + w)]
        return item

class Resize(torchvision.transforms.Resize):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, item):
        item['video'] = super().forward(item['video'])
        return item


class RGBSpatialCropSometimesUpscale(torch.nn.Module):
    '''This (randomly) crops the input video and with prob `sometimes_p` this crop is smaller but upscaled
    to `target_input_size`'''

    def __init__(self, sometimes_p, target_input_size, is_random, smaller_input_size=None):
        super().__init__()
        self.sometimes_p = sometimes_p
        self.do_sometimes_upscale = sometimes_p is not None and sometimes_p > 0

        self.crop_only = RGBSpatialCrop(target_input_size, is_random)

        if self.do_sometimes_upscale:
            self.crop_further_and_upscale = torchvision.transforms.Compose([
                RGBSpatialCrop(smaller_input_size, is_random),
                Resize(target_input_size),
            ])

    def forward(self, item):
        if self.do_sometimes_upscale and self.sometimes_p > torch.rand(1):
            return self.crop_further_and_upscale(item)
        else:
            return self.crop_only(item)


class RandomApplyColorDistortion(torch.nn.Module):

    def __init__(self, p_gray_scale=0., p_color_jitter=0., s=1.) -> None:
        super().__init__()
        self.p_gray_scale = p_gray_scale
        self.p_color_jitter = p_color_jitter
        self.s = s
        assert 0 <= self.p_color_jitter <= 1 and 0 <= self.p_gray_scale <= 1, (p_color_jitter, p_gray_scale)
        # SimCLR params
        color_jitter = torchvision.transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rand_color_jitter = torchvision.transforms.RandomApply([color_jitter], p_color_jitter)
        rand_gray = torchvision.transforms.RandomGrayscale(p_gray_scale)
        self.transforms = torchvision.transforms.Compose([rand_color_jitter, rand_gray])

    def forward(self, item):
        item['video'] = self.transforms(item['video'])
        return item

class ApplyColorJitterFrameWise(torch.nn.Module):

    def __init__(self, s=1.) -> None:
        super().__init__()
        self.s = s
        # SimCLR params
        self.transform = torchvision.transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)

    def forward(self, item):
        for i, frame in enumerate(item['video']):
            item['video'][i] = self.transform(frame)
        return item

class RandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):

    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(self, item):
        item['video'] = super().forward(item['video'])
        return item


def make_class_grid(leftmost_val, rightmost_val, grid_size, grid_type='linspace'):
    assert grid_size >= 3, f'grid_size: {grid_size} doesnot make sense. Look, 2 -> (-1,1); 1 -> (-1); 0 -> ()'
    if grid_type == 'linspace':
        return torch.from_numpy(np.linspace(leftmost_val, rightmost_val, grid_size)).float()
    elif grid_type == 'uniform':
        cell_size = (rightmost_val - leftmost_val) / grid_size
        cell_mid_point = cell_size / 2
        grid = np.linspace(leftmost_val - cell_mid_point, rightmost_val + cell_mid_point, grid_size + 2)[1:-1]
        return torch.from_numpy(grid).float()


def quantize_offset(grid: torch.Tensor, off_sec: float) -> Tuple[float, int]:
    '''Takes in the offset in seconds and snaps it onto the closest grid element.
    Returns the grid value and its index.'''
    closest_grid_el = (grid - off_sec).abs().argmin()
    return grid[closest_grid_el], closest_grid_el


class TemporalCropAndOffset(torch.nn.Module):
    '''
    During testing: we rely on the precalculated set of offsets and the position of the crop
                    from, e.g. visual track.

    If the offset prediction is formulated as classification, the offset in seconds is
    quantized to the closest number on the grid of desired size (`grid_size`)
    '''

    def __init__(self, crop_len_sec: float, max_off_sec: float, do_offset=True, grid_size: int = None):
        '''
        grid_size (int|None): if offset prediction is posed as classification, this is the number of classes
        '''
        super().__init__()
        self.crop_len_sec = crop_len_sec
        self.do_offset = do_offset
        self.grid_size = grid_size
        self.max_off_sec = max_off_sec
        if do_offset:
            self.class_grid = make_class_grid(-max_off_sec, max_off_sec, grid_size)

    def get_crop_idx(self, len_frames: int, crop_len_frames: int, is_random=True):
        if len_frames == crop_len_frames:
            return 0, len_frames
        if is_random:
            left_i = random.randint(0, len_frames - crop_len_frames)
        else:
            left_i = int(round((len_frames - crop_len_frames) / 2.))
        return left_i, left_i+crop_len_frames


class TemporalCropAndOffsetRandomFeasible(TemporalCropAndOffset):

    def __init__(self, crop_len_sec: float, max_off_sec: float, grid_type: str, way_to_do_trim: str = 'slice',
                 do_offset: bool = True, grid_size: int = None, max_wiggle_sec: float = None):
        super().__init__(crop_len_sec, max_off_sec, do_offset, grid_size)
        # TODO: rename crop to trim if temporal crop is used
        self.max_a_jitter_sec = max_wiggle_sec
        if do_offset:
            self.class_grid = make_class_grid(-max_off_sec, max_off_sec, grid_size, grid_type)
            logger.info(f'Offset class grid: {self.class_grid}')
            if self.max_a_jitter_sec is not None:
                assert max_wiggle_sec <= ((self.class_grid[1] - self.class_grid[0]) / 2), f'{self.class_grid}'
        self.grid_type = grid_type

    def apply_a_jitter(self, a_start_i, max_a_start_i, a_fps):
        max_a_jitter_i = sec2frames(self.max_a_jitter_sec, a_fps)  # not in self, as a_fps may be dynamic
        max_a_jitter_i_left = min(a_start_i, max_a_jitter_i)
        max_a_jitter_i_right = min(max_a_start_i - a_start_i, max_a_jitter_i)
        # jitter is U[left, right]
        a_jitter_i = random.randint(-max_a_jitter_i_left, max_a_jitter_i_right)
        # apply jitter
        a_start_i = a_start_i + a_jitter_i
        # making sure that any value from `a_start_i + U[left, right]` will be inside of [0, len-crop] region
        assert 0 <= a_start_i <= max_a_start_i, f'{a_jitter_i} {max_a_jitter_i_left} {max_a_jitter_i_right} {max_a_start_i}'
        return a_start_i, a_jitter_i


    def forward(self, item):
        vid = item['video']
        aud = item['audio']
        v_len_frames, C, H, W = vid.shape
        a_len_frames = aud.shape[0]

        v_fps = int(item['meta']['video']['fps'][0])
        a_fps = int(item['meta']['audio']['framerate'][0])

        v_crop_len_frames = sec2frames(self.crop_len_sec, v_fps)
        a_crop_len_frames = sec2frames(self.crop_len_sec, a_fps)

        if self.do_offset:
            # trying to get the offset parameters (for instance during valid and test we have fixed offsets)
            offset_sec = item['targets'].get('offset_sec', None)
            v_start_i_sec = item['targets'].get('v_start_i_sec', None)
            # train-time
            if offset_sec is None and v_start_i_sec is None:
                # aud starts `offset_sec` earlier than it should; aud has what will be shown after offset_sec
                offset_sec = random.choice(self.class_grid.tolist())
                offset_sec = round(offset_sec, 2)
                v_start_max_sec = frames2sec(v_len_frames - v_crop_len_frames, v_fps)
                # `v_start_sec` IS NOT rounded to the fps grid
                v_start_sec = random.uniform(max(0, -offset_sec), min(v_start_max_sec, v_start_max_sec-offset_sec))
                v_start_i = sec2frames(v_start_sec, v_fps)
                v_end_i = v_start_i + v_crop_len_frames
                # `v_start_i_sec` IS rounded to the fps grid
                v_start_i_sec = frames2sec(v_start_i, v_fps)
                # `a_start_i` depends on the rounded value `v_start_i_sec`, otherwise
                # (v_start_sec) we have ±0.1 jittering
                a_start_i = sec2frames(v_start_i_sec + offset_sec, a_fps)
                if self.max_a_jitter_sec is not None and self.max_a_jitter_sec > 0:
                    max_a_start_i = a_len_frames - a_crop_len_frames
                    a_start_i, a_jitter_i = self.apply_a_jitter(a_start_i, max_a_start_i, a_fps)
                    item['meta']['a_jitter_i'] = a_jitter_i
                a_end_i = a_start_i + a_crop_len_frames
            else:
                offset_sec = round(offset_sec, 2)
                v_start_i = sec2frames(v_start_i_sec, v_fps)
                a_start_i = sec2frames(v_start_i_sec + offset_sec, a_fps)
                v_end_i = v_start_i + v_crop_len_frames
                a_end_i = a_start_i + a_crop_len_frames
        else:
            offset_sec = 0.0
            is_random_crop = item['split'] == 'train'
            v_start_i, v_end_i = self.get_crop_idx(v_len_frames, v_crop_len_frames, is_random=is_random_crop)
            v_start_i_sec = frames2sec(v_start_i, v_fps)
            a_start_i = sec2frames(v_start_i_sec, a_fps)
            if self.max_a_jitter_sec is not None and self.max_a_jitter_sec > 0:
                max_a_start_i = a_len_frames - a_crop_len_frames
                a_start_i, a_jitter_i = self.apply_a_jitter(a_start_i, max_a_start_i, a_fps)
                item['meta']['a_jitter_i'] = a_jitter_i
            a_end_i = a_start_i + a_crop_len_frames

        # sometimes due to the rounding error e.g. v_start_sec = 1.505 but sec2frames(1.505, 25) = 1.48
        # given offset is -1.5, the a_start_i will be a small negative value. (likely a_fps * 1/v_fps * 0.5)
        if a_start_i < 0:
            how_much_out = a_start_i
            logger.info(f'a_start_i is negative ({how_much_out}) at {item["path"]}')
            if abs(how_much_out) <= a_fps / v_fps:
                logger.info('fixing it')
                a_start_i += abs(how_much_out)
                a_end_i += abs(how_much_out)
            else:
                raise Exception(f'{how_much_out} {item["path"]}')

        if aud.shape[0] < a_end_i:
            target_shape = list(aud.shape)
            target_shape[0] = a_end_i
            pad_aud = torch.zeros(target_shape)
            pad_aud[:aud.shape[0]] = aud
            aud = pad_aud
        if vid.shape[0] < v_end_i:
            target_shape = list(vid.shape)
            target_shape[0] = v_end_i
            pad_vid = torch.zeros(target_shape)
            pad_vid[:vid.shape[0]] = vid
            vid = pad_vid
        assert v_start_i < v_end_i and a_start_i < a_end_i
        assert aud.shape[0] >= a_end_i, f'{aud.shape} {a_end_i} {item["path"]}'
        assert vid.shape[0] >= v_end_i, f'{vid.shape} {v_end_i} {item["path"]}'

        vid, aud = vid[v_start_i:v_end_i, :, :, :], aud[a_start_i:a_end_i]

        item['video'] = vid
        item['audio'] = aud

        assert item['video'].shape[0] == v_fps * self.crop_len_sec, f'{item["video"].shape} {item["path"]}'
        assert item['audio'].shape[0] == a_fps * self.crop_len_sec, f'{item["audio"].shape} {item["path"]}'

        # caching parameters
        if self.do_offset:
            offset_label, offset_target = quantize_offset(self.class_grid, offset_sec)
            item['targets']['offset_sec'] = offset_sec
            item['targets']['v_start_i_sec'] = v_start_i_sec
            item['targets']['offset_label'] = offset_label
            item['targets']['offset_target'] = offset_target

        return item


class RGBToFloatToZeroOne(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, item):
        item['video'] = item['video'].to(torch.float32).div(255)
        return item


class RGBNormalize(torchvision.transforms.Normalize):
    '''The same as the torchvision`s but with different interface for the dict
    (C, H, W) or (B, C, H, W) to be normalized'''

    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)

    def forward(self, item):
        item['video'] = super().forward(item['video'])
        return item


class FreezeFrames(torch.nn.Module):
    '''Significant code overlap with CorruptAudio: TODO(refactor on a change)'''

    def __init__(self, max_off_sec, active_min_overlap_sec, to_freeze_frames) -> None:
        super().__init__()
        self.max_off_sec = max_off_sec
        self.active_min_overlap_sec = active_min_overlap_sec
        # making sure the kept RGB sequence has an overlap with audio track because tails of the offset
        # segment do not have either audio or visual infoamtion. So, if offset is 2, the active visual signal
        # is as long as 2 + active_min_overlap_sec
        self.sec_v_is_active = max_off_sec + active_min_overlap_sec
        self.to_freeze_frames = to_freeze_frames
        if to_freeze_frames:
            print(f'Going to freeze all frames except for {self.sec_v_is_active} sec randomly')

    def forward(self, item):
        # do nothing
        if not self.to_freeze_frames:
            return item

        # RGB: 123456789 -> 123|A|45|B|6789 -> 333456666
        vid = item['video']
        v_fps = item['meta']['video']['fps'][0]
        v_len_frames, C, H, W = vid.shape

        # pick active segment borders
        frames_v_is_active = sec2frames(self.sec_v_is_active, v_fps)
        start_frame = random.randint(0, v_len_frames-frames_v_is_active)
        # start_frame = int(v_fps*1.5)
        # print('start_frame', start_frame)
        # print('ALWAYS THE SAME')
        end_frame = start_frame + frames_v_is_active

        # the inactive segment on the left is replaced with the last frame (or left empty)
        frozen_left = vid[:start_frame][-1:].repeat((start_frame, 1, 1, 1))
        frozen_right = vid[end_frame:][:1].repeat((v_len_frames-end_frame, 1, 1, 1))

        assert len(frozen_left) + end_frame-start_frame + len(frozen_right) == len(vid), end_frame-start_frame

        vid = torch.cat([frozen_left, vid[start_frame:end_frame], frozen_right])
        item['video'] = vid

        return item

class CorruptAudio(torch.nn.Module):
    '''Significant code overlap with Freeze Frames: TODO(refactor on a change)'''

    def __init__(self, max_off_sec, active_min_overlap_sec, to_corrupt_audio, corrupt_type) -> None:
        super().__init__()
        self.max_off_sec = max_off_sec
        self.active_min_overlap_sec = active_min_overlap_sec
        self.to_corrupt_audio = to_corrupt_audio
        self.corrupt_type = corrupt_type
        # making sure the kept signal has an overlap with RGB track because otherwise tails of the offset
        # segment would not have either audio or visual infoamtion. So, if offset is 2, the active
        # visual signal is as long as 2 + active_min_overlap_sec
        self.sec_is_active = max_off_sec + active_min_overlap_sec
        if to_corrupt_audio:
            assert corrupt_type in ['rand', 'mute'], f'corrupt type {corrupt_type} is not supported'
            print(f'Going to corrupt audio except for {self.sec_is_active} sec randomly')

    def forward(self, item):
        # do nothing
        if not self.to_corrupt_audio:
            return item

        aud = item['audio']
        a_fps = item['meta']['audio']['framerate'][0]
        a_len_frames = len(aud)
        frames_is_active = sec2frames(self.sec_is_active, a_fps)
        start_frame = random.randint(0, a_len_frames-frames_is_active)
        # start_frame = int(a_fps * 1.5)
        # print('start_frame', start_frame)
        # print('ALWAYS THE SAME')
        end_frame = start_frame + frames_is_active

        # audio: 123456789 -> 123|A|45|B|6789 -> hbi45bxua (if corrupt_type = 'rand') or 000450000 (if 'mute')
        if self.corrupt_type == 'mute':
            frozen_left = torch.zeros(start_frame, dtype=aud.dtype)
            frozen_right = torch.zeros(a_len_frames-end_frame, dtype=aud.dtype)
        elif self.corrupt_type == 'rand':
            frozen_left = torch.rand(start_frame, dtype=aud.dtype) * 2 - 1  # to [-1, 1]
            frozen_right = torch.rand(a_len_frames-end_frame, dtype=aud.dtype) * 2 - 1  # to [-1, 1]

        assert len(frozen_left) + end_frame-start_frame + len(frozen_right) == len(aud), end_frame-start_frame

        aud = torch.cat([frozen_left, aud[start_frame:end_frame], frozen_right])

        item['audio'] = aud

        return item


class RGBTakeFirstFrame(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        print('TAKING ONLY THE FIRST FRAME')
        print('TAKING ONLY THE FIRST FRAME')
        print('TAKING ONLY THE FIRST FRAME')
        print('TAKING ONLY THE FIRST FRAME')

    def forward(self, item):
        # square brackets to keep the dim
        item['video'] = item['video'][[0]]
        return item


class AudioTrimOrTileToDuration(torch.nn.Module):
    '''only used during audio feature extractor pre-training'''

    def __init__(self, duration_sec: int = 10) -> None:
        super().__init__()
        self.duration_sec = duration_sec
        assert duration_sec == int(duration_sec), 'please use int as duration_sec, otherwise change tiling'

    def forward(self, item):
        samples = item['audio']
        meta = item['meta']
        sr = meta['audio']['framerate'][0]
        target_len_frames = int(sr * self.duration_sec)
        samples = torch.tile(samples, (self.duration_sec, ))[:target_len_frames]
        item['audio'] = samples
        item['meta']['new_duration'] = self.duration_sec
        return item


class AudioTimeCrop(torch.nn.Module):
    '''only used during audio feature extractor pre-training'''

    def __init__(self, crop_len_sec, is_random) -> None:
        super().__init__()
        self.crop_len_sec = crop_len_sec
        self.is_random = is_random

    @staticmethod
    def sec2frames(sec, fps):
        return int(sec * fps)

    def get_random_crop(self, len_frames: int, crop_len_frames: int):
        if len_frames == crop_len_frames:
            return 0, len_frames
        left_i = random.randint(0, len_frames - crop_len_frames)
        return left_i, left_i+crop_len_frames

    def get_center_crop(self, len_frames, crop_len_frames):
        left_i = int(round((len_frames - crop_len_frames) / 2.))
        return left_i, left_i+crop_len_frames

    def forward(self, item):
        aud = item['audio']
        a_len_frames = aud.shape[0]
        a_fps = int(item['meta']['audio']['framerate'][0])
        assert a_fps == item['meta']['audio']['framerate'][0], f'{a_fps} ' + \
            item['meta']['audio']['framerate'][0]
        a_crop_len_frames = sec2frames(self.crop_len_sec, a_fps)
        if self.is_random:
            a_left_i, a_right_i = self.get_random_crop(a_len_frames, a_crop_len_frames)
        else:
            a_left_i, a_right_i = self.get_center_crop(a_len_frames, a_crop_len_frames)
        item['audio'] = aud[a_left_i:a_right_i]
        item['meta']['new_duration'] = self.crop_len_sec
        return item


class AudioRandomVolume(torch.nn.Module):

    def __init__(self, p: float, **kwargs):
        super().__init__()
        transform = torchaudio.transforms.Vol(**kwargs)
        self.transform = torchvision.transforms.RandomApply([transform], p)

    def forward(self, item):
        item['audio'] = self.transform(item['audio'])
        return item

class AudioRandomLowpassFilter(torch.nn.Module):

    def __init__(self, p: float, cutoff_freq: float, Q: float = 0.707):
        super().__init__()
        self.p = p
        self.cutoff_freq = cutoff_freq
        self.Q = Q

    def forward(self, item):
        if self.p > torch.rand(1):
            sr = int(item['meta']['audio']['framerate'][0])
            wave = item['audio'].unsqueeze(0)
            wave = torchaudio.functional.lowpass_biquad(wave, sr, self.cutoff_freq, self.Q)
            item['audio'] = wave.squeeze(0)
        return item

class AudioRandomPitchShift(torch.nn.Module):

    def __init__(self, p: float, shift: int) -> None:
        super().__init__()
        self.p = p
        self.shift = shift

    def forward(self, item):
        if self.p > torch.rand(1):
            sr = int(item['meta']['audio']['framerate'][0])
            effects = [['pitch', f'{self.shift}'], ['rate', f'{sr}']]
            wave = item['audio'].unsqueeze(0)
            wave, _ = torchaudio.sox_effects.apply_effects_tensor(wave, sr, effects)
            item['audio'] = wave.squeeze(0)
        return item

class AudioRandomReverb(torch.nn.Module):

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p
        self.effects = [['reverb', '-w']]

    def forward(self, item):
        if self.p > torch.rand(1):
            sr = int(item['meta']['audio']['framerate'][0])
            wave = item['audio'].unsqueeze(0)
            item['audio'], _ = torchaudio.sox_effects.apply_effects_tensor(wave, sr, self.effects)
            item['audio'] = item['audio'].mean(dim=0)
        return item

class AudioRandomGaussNoise(torch.nn.Module):

    def __init__(self, p: float, amplitude=0.01) -> None:
        super().__init__()
        self.p = p
        self.amplitude = amplitude

    def forward(self, item):
        if self.p > torch.rand(1):
            wave = item['audio']
            noise = torch.randn_like(wave, dtype=wave.dtype)
            item['audio'] = wave + self.amplitude * noise
        return item


class AudioSpectrogram(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.spec = torchaudio.transforms.Spectrogram(**kwargs)

    def forward(self, item):
        item['audio'] = self.spec(item['audio'])
        return item


class AudioRandomFreqMask(torch.nn.Module):

    def __init__(self, p: float, freq_mask_param: int, iid_masks: bool = False):
        super().__init__()
        transform = torchaudio.transforms.FrequencyMasking(freq_mask_param, iid_masks)
        self.transform = torchvision.transforms.RandomApply([transform], p)

    def forward(self, item):
        item['audio'] = self.transform(item['audio'])
        return item

class AudioRandomTimeMask(torch.nn.Module):

    def __init__(self, p: float, time_mask_param: int, iid_masks: bool = False, ratio: float = 1.0):
        super().__init__()
        transform = torchaudio.transforms.TimeMasking(time_mask_param, iid_masks, ratio)
        self.transform = torchvision.transforms.RandomApply([transform], p)

    def forward(self, item):
        # for some reason TimeMasking (compared to FrequencyMasking) requires a 3d tensor
        item['audio'] = self.transform(item['audio'].unsqueeze(0)).squeeze(0)
        return item

class AudioLog(torch.nn.Module):

    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, item):
        item['audio'] = torch.log(item['audio'] + self.eps)
        return item


class AudioStandardNormalize(torch.nn.Module):
    '''Normalization is done frequency wise (means.shape and stds.shape = F)'''

    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, item):
        aud = item['audio']
        means = item['audio'].mean(dim=1, keepdim=True)
        stds = item['audio'].std(dim=1, keepdim=True)
        item['audio'] = (aud - means) / (stds + self.eps)
        item['meta']['spec_means'] = means.squeeze()
        item['meta']['spec_stds'] = stds.squeeze()
        return item


class AudioUnsqueezeChannelDim(torch.nn.Module):

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, item):
        item['audio'] = item['audio'].unsqueeze(self.dim)
        return item


if __name__ == '__main__':
    grid = make_class_grid(-1, 1, 21, grid_type='linspace')
    grid = make_class_grid(-2, 2, 41, grid_type='uniform')
    print('grid:', grid)
    print('value quantization:', quantize_offset(grid, 0.06))
    v_fps = 25.0
    duration = 9.0

    input = {
        'video': torch.randint(0, 256, (int(duration * v_fps), 3, 720//2, 1280//2), dtype=torch.uint8),
        'audio': torch.arange(221184-1).float(),
        'targets': {},
        'meta': {
            'video': {'duration': [duration], 'fps': [v_fps]},
            'audio': {'duration': [duration], 'framerate': [22050.0]},
            'subtitles': {'duration': []},
            'cc': {'duration': []},
        },
        'path': '/home/nvme/data/vggsound/video/-5cWCaoEDlE_261000_271000.mp4',
        'split': 'train',
    }

    print(input['audio'].shape, input['video'].shape)

    fn = AudioResampleDynamic(22050)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = EqualifyFromRight(clip_max_len_sec=10)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape)

    # fn = RGBSpatialCrop((224, 224), is_random=True)
    fn = RGBSpatialCrop((112, 112), is_random=True)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = Resize((224, 224))
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = RandomApplyColorDistortion(p_gray_scale=0.5, p_color_jitter=0.5, s=1.0)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = TemporalCropAndOffsetRandomFeasible(crop_len_sec=5, max_off_sec=2, do_offset=True, grid_size=3,
                                             grid_type='linspace', max_wiggle_sec=0.3)
    fn = TemporalCropAndOffsetRandomFeasible(crop_len_sec=5, max_off_sec=2, do_offset=True, grid_size=3,
                                             grid_type='linspace')
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])
    print(input['targets'])
    print(input['meta'])

    fn = FreezeFrames(max_off_sec=2, active_min_overlap_sec=1, to_freeze_frames=True)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = CorruptAudio(max_off_sec=2, active_min_overlap_sec=1, to_corrupt_audio=True, corrupt_type='mute')
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = RGBToFloatToZeroOne()
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])
    print(input['meta'])

    fn = RGBNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])
    print(input['video'].mean(dim=(0, 2, 3)))
    print(input['meta'])

    fn = AudioRandomReverb(p=1.0)
    input = fn(input)

    fn = AudioSpectrogram(n_fft=512, hop_length=512//4)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = AudioLog()
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = AudioStandardNormalize()
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = AudioUnsqueezeChannelDim(dim=0)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    # audio only
    input = {
        'audio': torch.arange(221184).float(),
        'meta': {
            'video': {'duration': [10.0], 'fps': [10.0]},
            'audio': {'duration': [11.0], 'framerate': [22050.0]},
            'subtitles': {'duration': []},
            'cc': {'duration': []}
        },
        'path': '/home/nvme/data/vggsound/video/-5cWCaoEDlE_261000_271000.mp4'
    }

    print(input['audio'].shape)

    fn = AudioTrimOrTileToDuration(duration_sec=10)
    input = fn(input)
    print(input['audio'].shape, input['meta']['audio'])
    print(input['meta'])

    fn = AudioTimeCrop(crop_len_sec=8, is_random=True)
    input = fn(input)
    print(input['audio'].shape, input['meta']['audio'])
    print(input['meta'])

    fn = AudioTimeCrop(crop_len_sec=8, is_random=False)
    input = fn(input)
    print(input['audio'].shape, input['meta']['audio'])
    print(input['meta'])

    fn = AudioSpectrogram(n_fft=512, hop_length=512//4)
    input = fn(input)
    print(input['audio'].shape, input['meta']['audio'])
    print(input['meta'])

    fn = AudioLog()
    input = fn(input)
    print(input['audio'].shape, input['meta']['audio'])
    print(input['meta'])
    print(input['audio'].min(), input['audio'].max())

    fn = AudioStandardNormalize()
    input = fn(input)
    print(input['audio'].shape, input['meta']['audio'])
    print(input['meta'])
