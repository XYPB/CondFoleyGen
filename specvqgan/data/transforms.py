import torch
import torchaudio
import torchaudio.functional
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import math
import random
import soundfile
import os
import librosa
import albumentations
from torch_pitch_shift import *

SR = 22050

class ResizeShortSide(object):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, x):
        '''
        x must be PIL.Image
        '''
        w, h = x.size
        short_side = min(w, h)
        w_target = int((w / short_side) * self.size)
        h_target = int((h / short_side) * self.size)
        return x.resize((w_target, h_target))


class Crop(object):
    def __init__(self, cropped_shape=None, random_crop=False):
        self.cropped_shape = cropped_shape
        if cropped_shape is not None:
            mel_num, spec_len = cropped_shape
            if random_crop:
                self.cropper = albumentations.RandomCrop
            else:
                self.cropper = albumentations.CenterCrop
            self.preprocessor = albumentations.Compose([self.cropper(mel_num, spec_len)])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __call__(self, item):
        item['image'] = self.preprocessor(image=item['image'])['image']
        if 'cond_image' in item.keys():
            item['cond_image'] = self.preprocessor(image=item['cond_image'])['image']
        return item

class CropImage(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

class CropFeats(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

    def __call__(self, item):
        item['feature'] = self.preprocessor(image=item['feature'])['image']
        return item

class CropCoords(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

    def __call__(self, item):
        item['coord'] = self.preprocessor(image=item['coord'])['image']
        return item


class RandomResizedCrop3D(nn.Module):
    """Crop the given series of images to random size and aspect ratio.
    The image can be a PIL Images or a Tensor, in which case it is expected
    to have [N, ..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
      size (int or sequence): expected output size of each edge. If size is an
        int instead of sequence like (h, w), a square output size ``(size, size)`` is
        made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
      scale (tuple of float): range of size of the origin size cropped
      ratio (tuple of float): range of aspect ratio of the origin aspect ratio cropped.
      interpolation (int): Desired interpolation enum defined by `filters`_.
        Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
        and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=transforms.InterpolationMode.BILINEAR):
        super().__init__()
        if isinstance(size, tuple) and len(size) == 2:
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
          img (PIL Image or Tensor): Input image.
          scale (list): range of scale of the origin size cropped
          ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
          tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = img.size
        area = height * width

        for _ in range(10):
            target_area = area * \
                torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, imgs):
        """
        Args:
          img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
          PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in imgs]


class Resize3D(object):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, imgs):
        '''
        x must be PIL.Image
        '''
        return [x.resize((self.size, self.size)) for x in imgs]


class RandomHorizontalFlip3D(object):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, imgs):
        '''
        x must be PIL.Image
        '''
        if np.random.rand() < self.p:
            return [x.transpose(Image.FLIP_LEFT_RIGHT) for x in imgs]
        else:
            return imgs


class ColorJitter3D(torch.nn.Module):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
    brightness (float or tuple of float (min, max)): How much to jitter brightness.
        brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
        or the given [min, max]. Should be non negative numbers.
    contrast (float or tuple of float (min, max)): How much to jitter contrast.
        contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
        or the given [min, max]. Should be non negative numbers.
    saturation (float or tuple of float (min, max)): How much to jitter saturation.
        saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
        or the given [min, max]. Should be non negative numbers.
    hue (float or tuple of float (min, max)): How much to jitter hue.
        hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
        Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = (1-brightness, 1+brightness)
        self.contrast = (1-contrast, 1+contrast)
        self.saturation = (1-saturation, 1+saturation)
        self.hue = (0-hue, 0+hue)

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        tfs = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            tfs.append(transforms.Lambda(
                lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            tfs.append(transforms.Lambda(
                lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            tfs.append(transforms.Lambda(
                lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            tfs.append(transforms.Lambda(
                lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(tfs)
        transform = transforms.Compose(tfs)

        return transform

    def forward(self, imgs):
        """
        Args:
          img (PIL Image or Tensor): Input image.

        Returns:
          PIL Image or Tensor: Color jittered image.
        """
        transform = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
        return [transform(img) for img in imgs]


class ToTensor3D(object):
    def __init__(self):
        super().__init__()

    def __call__(self, imgs):
        '''
        x must be PIL.Image
        '''
        return [F.to_tensor(img) for img in imgs]


class Normalize3D(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, imgs):
        '''
        x must be PIL.Image
        '''
        return [F.normalize(img, self.mean, self.std, self.inplace) for img in imgs]


class CenterCrop3D(object):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, imgs):
        '''
        x must be PIL.Image
        '''
        return [F.center_crop(img, self.size) for img in imgs]


class FrequencyMasking(object):
    def __init__(self, freq_mask_param: int, iid_masks: bool = False):
        super().__init__()
        self.masking = torchaudio.transforms.FrequencyMasking(freq_mask_param, iid_masks)

    def __call__(self, item):
        if 'cond_image' in item.keys():
            batched_spec = torch.stack(
                [torch.tensor(item['image']), torch.tensor(item['cond_image'])], dim=0
            )[:, None] # (2, 1, H, W)
            masked = self.masking(batched_spec).numpy()
            item['image'] = masked[0, 0]
            item['cond_image'] = masked[1, 0]
        elif 'image' in item.keys():
            inp = torch.tensor(item['image'])
            item['image'] = self.masking(inp).numpy()
        else:
            raise NotImplementedError()
        return item


class TimeMasking(object):
    def __init__(self, time_mask_param: int, iid_masks: bool = False):
        super().__init__()
        self.masking = torchaudio.transforms.TimeMasking(time_mask_param, iid_masks)

    def __call__(self, item):
        if 'cond_image' in item.keys():
            batched_spec = torch.stack(
                [torch.tensor(item['image']), torch.tensor(item['cond_image'])], dim=0
            )[:, None] # (2, 1, H, W)
            masked = self.masking(batched_spec).numpy()
            item['image'] = masked[0, 0]
            item['cond_image'] = masked[1, 0]
        elif 'image' in item.keys():
            inp = torch.tensor(item['image'])
            item['image'] = self.masking(inp).numpy()
        else:
            raise NotImplementedError()
        return item


class PitchShift(nn.Module):

    def __init__(self, up=12, down=-12, sample_rate=SR):
        super().__init__()
        self.range = (down, up)
        self.sr = sample_rate

    def forward(self, x):
        assert len(x.shape) == 2
        x = x[:, None, :]
        ratio = float(random.randint(self.range[0], self.range[1]) / 12.)
        shifted = pitch_shift(x, ratio, self.sr)
        return shifted.squeeze()


class MelSpectrogram(object):
    def __init__(self, sr, nfft, fmin, fmax, nmels, hoplen, spec_power, inverse=False):
        self.sr = sr
        self.nfft = nfft
        self.fmin = fmin
        self.fmax = fmax
        self.nmels = nmels
        self.hoplen = hoplen
        self.spec_power = spec_power
        self.inverse = inverse

        self.mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, fmin=fmin, fmax=fmax, n_mels=nmels)

    def __call__(self, x):
        x = x.numpy()
        if self.inverse:
            spec = librosa.feature.inverse.mel_to_stft(
                x, sr=self.sr, n_fft=self.nfft, fmin=self.fmin, fmax=self.fmax, power=self.spec_power
            )
            wav = librosa.griffinlim(spec, hop_length=self.hoplen)
            return torch.FloatTensor(wav)
        else:
            spec = np.abs(librosa.stft(x, n_fft=self.nfft, hop_length=self.hoplen)) ** self.spec_power
            mel_spec = np.dot(self.mel_basis, spec)
            return torch.FloatTensor(mel_spec)

class SpectrogramTorchAudio(object):
    def __init__(self, nfft, hoplen, spec_power, inverse=False):
        self.nfft = nfft
        self.hoplen = hoplen
        self.spec_power = spec_power
        self.inverse = inverse

        self.spec_trans = torchaudio.transforms.Spectrogram(
            n_fft=self.nfft,
            hop_length=self.hoplen,
            power=self.spec_power,
        )
        self.inv_spec_trans = torchaudio.transforms.GriffinLim(
            n_fft=self.nfft,
            hop_length=self.hoplen,
            power=self.spec_power,
        )

    def __call__(self, x):
        if self.inverse:
            wav = self.inv_spec_trans(x)
            return wav
        else:
            spec = torch.abs(self.spec_trans(x))
            return spec


class MelScaleTorchAudio(object):
    def __init__(self, sr, stft, fmin, fmax, nmels, inverse=False):
        self.sr = sr
        self.stft = stft
        self.fmin = fmin
        self.fmax = fmax
        self.nmels = nmels
        self.inverse = inverse

        self.mel_trans = torchaudio.transforms.MelScale(
            n_mels=self.nmels,
            sample_rate=self.sr,
            f_min=self.fmin,
            f_max=self.fmax,
            n_stft=self.stft,
            norm='slaney'
        )
        self.inv_mel_trans = torchaudio.transforms.InverseMelScale(
            n_mels=self.nmels,
            sample_rate=self.sr,
            f_min=self.fmin,
            f_max=self.fmax,
            n_stft=self.stft,
            norm='slaney'
        )

    def __call__(self, x):
        if self.inverse:
            spec = self.inv_mel_trans(x)
            return spec
        else:
            mel_spec = self.mel_trans(x)
            return mel_spec

class Padding(object):
    def __init__(self, target_len, inverse=False):
        self.target_len=int(target_len)
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            x = x.squeeze()
            if x.shape[0] < self.target_len:
                pad = torch.zeros((self.target_len,), dtype=x.dtype, device=x.device)
                pad[:x.shape[0]] = x
                x = pad
            elif x.shape[0] > self.target_len:
                raise NotImplementedError()
            return x

class MakeMono(object):
    def __init__(self, inverse=False):
        self.inverse = inverse
    
    def __call__(self, x):
        if self.inverse:
            return x
        else:
            x = x.squeeze()
            if len(x.shape) == 1:
                return torch.FloatTensor(x)
            elif len(x.shape) == 2:
                target_dim = int(torch.argmin(torch.tensor(x.shape)))
                return torch.mean(x, dim=target_dim)
            else:
                raise NotImplementedError

class LowerThresh(object):
    def __init__(self, min_val, inverse=False):
        self.min_val = torch.tensor(min_val)
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return torch.maximum(self.min_val, x)

class Add(object):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x - self.val
        else:
            return x + self.val

class Subtract(Add):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x + self.val
        else:
            return x - self.val

class Multiply(object):
    def __init__(self, val, inverse=False) -> None:
        self.val = val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x / self.val
        else:
            return x * self.val

class Divide(Multiply):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x * self.val
        else:
            return x / self.val


class Log10(object):
    def __init__(self, inverse=False):
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return 10 ** x
        else:
            return torch.log10(x)

class Clip(object):
    def __init__(self, min_val, max_val, inverse=False):
        self.min_val = min_val
        self.max_val = max_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return torch.clip(x, self.min_val, self.max_val)

class TrimSpec(object):
    def __init__(self, max_len, inverse=False):
        self.max_len = max_len
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x[:, :self.max_len]

class MaxNorm(object):
    def __init__(self, inverse=False):
        self.inverse = inverse
        self.eps = 1e-10

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x / (x.max() + self.eps)


class NormalizeAudio(object):
    def __init__(self, inverse=False, desired_rms=0.1, eps=1e-4):
        self.inverse = inverse
        self.desired_rms = desired_rms
        self.eps = torch.tensor(eps)

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            rms = torch.maximum(self.eps, torch.sqrt(torch.mean(x**2)))
            x = x * (self.desired_rms / rms)
            x[x > 1.] = 1.
            x[x < -1.] = -1.
            return x


class RandomNormalizeAudio(object):
    def __init__(self, inverse=False, rms_range=[0.05, 0.2], eps=1e-4):
        self.inverse = inverse
        self.rms_low, self.rms_high = rms_range
        self.eps = torch.tensor(eps)

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            rms = torch.maximum(self.eps, torch.sqrt(torch.mean(x**2)))
            desired_rms = (torch.rand(1) * (self.rms_high - self.rms_low)) + self.rms_low
            x = x * (desired_rms / rms)
            x[x > 1.] = 1.
            x[x < -1.] = -1.
            return x


class MakeDouble(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.to(torch.double)


class MakeFloat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.to(torch.float)


class Wave2Spectrogram(nn.Module):
    def __init__(self, mel_num, spec_crop_len):
        super().__init__()
        self.trans = transforms.Compose([
            LowerThresh(1e-5),
            Log10(),
            Multiply(20),
            Subtract(20),
            Add(100),
            Divide(100),
            Clip(0, 1.0),
            TrimSpec(173),
            transforms.CenterCrop((mel_num, spec_crop_len))
        ])

    def forward(self, x):
        return self.trans(x)



TRANSFORMS = transforms.Compose([
    SpectrogramTorchAudio(nfft=1024, hoplen=1024//4, spec_power=1),
    MelScaleTorchAudio(sr=22050, stft=513, fmin=125, fmax=7600, nmels=80),
    LowerThresh(1e-5),
    Log10(),
    Multiply(20),
    Subtract(20),
    Add(100),
    Divide(100),
    Clip(0, 1.0),
])

def get_spectrogram_torch(audio_path, save_dir, length, save_results=True):
    wav, _ = soundfile.read(audio_path)
    wav = torch.FloatTensor(wav)
    y = torch.zeros(length)
    if wav.shape[0] < length:
        y[:len(wav)] = wav
    else:
        y = wav[:length]
    
    mel_spec = TRANSFORMS(y).numpy()
    y = y.numpy()
    if save_results:
        os.makedirs(save_dir, exist_ok=True)
        audio_name = os.path.basename(audio_path).split('.')[0]
        np.save(os.path.join(save_dir, audio_name + '_mel.npy'), mel_spec)
        np.save(os.path.join(save_dir, audio_name + '_audio.npy'), y)
    else:
        return y, mel_spec
