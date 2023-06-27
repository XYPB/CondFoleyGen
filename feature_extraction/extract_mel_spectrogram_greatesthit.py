import argparse
import os
import os.path as P
from copy import deepcopy
from functools import partial
from glob import glob
from multiprocessing import Pool
from pathlib import Path
import json

import librosa
import numpy as np
import torchvision
import torchaudio
import torch


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
        if self.inverse:
            spec = librosa.feature.inverse.mel_to_stft(
                x, sr=self.sr, n_fft=self.nfft, fmin=self.fmin, fmax=self.fmax, power=self.spec_power
            )
            wav = librosa.griffinlim(spec, hop_length=self.hoplen)
            return wav
        else:
            spec = np.abs(librosa.stft(x, n_fft=self.nfft, hop_length=self.hoplen)) ** self.spec_power
            mel_spec = np.dot(self.mel_basis, spec)
            return mel_spec


class MelSpectrogramTorchAudio(object):
    def __init__(self, sr, nfft, fmin, fmax, nmels, hoplen, spec_power, inverse=False):
        self.sr = sr
        self.nfft = nfft
        self.fmin = fmin
        self.fmax = fmax
        self.nmels = nmels
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
        self.mel_trans = torchaudio.transforms.MelScale(
            n_mels=self.nmels,
            sample_rate=self.sr,
            f_min=self.fmin,
            f_max=self.fmax,
            n_stft=self.nfft,
            mel_scale='salney',
            norm='salney'
        )
        self.inv_mel_trans = torchaudio.transforms.InverseMelScale(
            n_mels=self.nmels,
            sample_rate=self.sr,
            f_min=self.fmin,
            f_max=self.fmax,
            n_stft=self.nfft,
            mel_scale='salney',
            norm='salney'
        )

    def __call__(self, x):
        if self.inverse:
            spec = self.inv_mel_trans(x)
            wav = self.inv_spec_trans(spec)
            return wav
        else:
            spec = torch.abs(self.spec_trans(x))
            mel_spec = self.mel_trans(spec)
            return mel_spec


class LowerThresh(object):
    def __init__(self, min_val, inverse=False):
        self.min_val = min_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.maximum(self.min_val, x)

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
            return np.log10(x)

class Clip(object):
    def __init__(self, min_val, max_val, inverse=False):
        self.min_val = min_val
        self.max_val = max_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.clip(x, self.min_val, self.max_val)

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
        self.eps = eps

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            rms = np.maximum(self.eps, np.sqrt(np.mean(x**2)))
            x = x * (self.desired_rms / rms)
            x[x > 1.] = 1.
            x[x < -1.] = -1.
            return x


class RandomNormalizeAudio(object):
    def __init__(self, inverse=False, rms_range=[0.05, 0.2], eps=1e-4):
        self.inverse = inverse
        self.rms_low, self.rms_high = rms_range
        self.eps = eps

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            rms = np.maximum(self.eps, np.sqrt(np.mean(x**2)))
            desired_rms = (torch.rand(1) * (self.rms_high - self.rms_low)) + self.rms_low
            x = x * (desired_rms / rms)
            x[x > 1.] = 1.
            x[x < -1.] = -1.
            return x


TRANSFORMS = torchvision.transforms.Compose([
    NormalizeAudio(desired_rms=0.1, eps=1e-4),
    MelSpectrogram(sr=22050, nfft=1024, fmin=125, fmax=7600, nmels=80, hoplen=1024//4, spec_power=1),
    LowerThresh(1e-5),
    Log10(),
    Multiply(20),
    Subtract(20),
    Add(100),
    Divide(100),
    Clip(0, 1.0),
    TrimSpec(860)
])

SR = 22050

def inv_transforms(x, folder_name='melspec_10s_22050hz'):
    '''relies on the GLOBAL contant TRANSFORMS which should be defined in this document'''
    if folder_name == 'melspec_10s_22050hz':
        i_transforms = deepcopy(TRANSFORMS.transforms[::-1])
    else:
        raise NotImplementedError
    for t in i_transforms:
        t.inverse = True
    i_transforms = torchvision.transforms.Compose(i_transforms)
    return i_transforms(x)


def get_spectrogram_infer(audio_path, save_dir, length, folder_name='melspec_10s_22050hz', save_results=True):
    print("Using normalized spec")
    wav, _ = librosa.load(audio_path, sr=None)
    # this cannot be a transform without creating a huge overhead with inserting audio_name in each
    y = np.zeros(length)
    if wav.shape[0] < length:
        y[:len(wav)] = wav
    else:
        y = wav[:length]

    if folder_name == 'melspec_10s_22050hz':
        print('using', folder_name)
        mel_spec = TRANSFORMS(y)
    else:
        raise NotImplementedError

    if save_results:
        os.makedirs(save_dir, exist_ok=True)
        audio_name = os.path.basename(audio_path).split('.')[0]
        np.save(P.join(save_dir, audio_name + '_mel.npy'), mel_spec)
        np.save(P.join(save_dir, audio_name + '_audio.npy'), y)
    else:
        return y, mel_spec


def get_spectrogram(audio_info, save_dir, length, folder_name='melspec_10s_22050hz', save_results=True):
    audio_path, t, hit_type = audio_info
    wav, _ = librosa.load(audio_path, sr=SR)
    s_idx, e_idx = int((t - 0.5) * SR), int((t + 1.5) * SR)
    e_idx = e_idx if e_idx < len(wav) else len(wav)
    # this cannot be a transform without creating a huge overhead with inserting audio_name in each
    y = np.zeros(length)
    if s_idx + length > len(wav):
        y[:e_idx-s_idx] = wav[s_idx:e_idx]
    else:
        y = wav[s_idx:s_idx+length]

    if folder_name == 'melspec_10s_22050hz':
        print('using', folder_name)
        mel_spec = TRANSFORMS(y)
    else:
        raise NotImplementedError

    if save_results:
        os.makedirs(save_dir, exist_ok=True)
        audio_name = os.path.basename(audio_path).split('.')[0]
        np.save(P.join(save_dir, audio_name + f'_{s_idx}_' + hit_type + '_mel.npy'), mel_spec)
        np.save(P.join(save_dir, audio_name + f'_{s_idx}_' + hit_type + '_audio.npy'), y)
    else:
        return y, mel_spec


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument("-i", "--input_dir", default="data/greatesthit/greatesthit_orig")
    paser.add_argument("-i2", "--input_dir2", default="data/greatesthit/greatesthit_processed")
    paser.add_argument("-o", "--output_dir", default="data/greatesthit/melspec_10s_22050hz")
    paser.add_argument("-l", "--length", default=220500)
    paser.add_argument("-n", '--num_worker', type=int, default=8)
    args = paser.parse_args()
    input_dir = args.input_dir
    input_dir2 = args.input_dir2
    output_dir = args.output_dir
    length = args.length

    audio_paths = glob(P.join(input_dir, "*_denoised.wav"))
    audio_paths.sort()
    audio_infos = []
    for ap in audio_paths:
        t_path = ap.replace(input_dir, input_dir2).replace('_denoised.wav', '')
        ts = json.load(open(P.join(t_path, 'hit_record.json'), 'r'))
        for t, hit_type in ts:
            if t < 0.5:
                continue
            audio_infos.append([ap, t, hit_type.replace(' ', '_')])

    with Pool(args.num_worker) as p:
        p.map(partial(
            get_spectrogram, save_dir=output_dir, length=length, folder_name=Path(output_dir).name
        ), audio_infos)
