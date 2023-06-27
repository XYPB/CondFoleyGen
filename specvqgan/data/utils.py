import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from random import shuffle, choice, sample

from moviepy.editor import VideoFileClip
import librosa
from scipy import signal
from scipy.io import wavfile
import torchaudio
torchaudio.set_audio_backend("sox_io")

INTERVAL = 1000

# discard
stft = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, hop_length=161, n_mels=64).cuda()


def log10(x): return torch.log(x)/torch.log(torch.tensor(10.))


def norm_range(x, min_val, max_val):
    return 2.*(x - min_val)/float(max_val - min_val) - 1.


def normalize_spec(spec, spec_min, spec_max):
    return norm_range(spec, spec_min, spec_max)


def db_from_amp(x, cuda=False):
    # rescale the audio
    if cuda:
        return 20. * log10(torch.max(torch.tensor(1e-5).to('cuda'), x.float()))
    else:
        return 20. * log10(torch.max(torch.tensor(1e-5), x.float()))


def audio_stft(audio, stft=stft):
    # We'll apply stft to the audio samples to convert it to a HxW matrix
    N, C, A = audio.size()
    audio = audio.view(N * C, A)
    spec = stft(audio)
    spec = spec.transpose(-1, -2)
    spec = db_from_amp(spec, cuda=True)
    spec = normalize_spec(spec, -100., 100.)
    _, T, F = spec.size()
    spec = spec.view(N, C, T, F)
    return spec


# discard
# def get_spec(
#     wavs,
#     sample_rate=16000,
#     use_volume_jittering=False,
#     center=False,
# ):
#     # Volume  jittering - scale volume by factor in range (0.9, 1.1)
#     if use_volume_jittering:
#         wavs = [wav * np.random.uniform(0.9, 1.1) for wav in wavs]
#     if center:
#         wavs = [center_only(wav) for wav in wavs]

#     # Convert to log filterbank
#     specs = [logfbank(
#         wav,
#         sample_rate,
#         winlen=0.009,
#         winstep=0.005,  # if num_sec==1 else 0.01,
#         nfilt=256,
#         nfft=1024
#     ).astype('float32').T for wav in wavs]

#     # Convert to 32-bit float and expand dim
#     specs = np.stack(specs, axis=0)
#     specs = np.expand_dims(specs, 1)
#     specs = torch.as_tensor(specs)  # Nx1xFxT

#     return specs


def center_only(audio, sr=16000, L=1.0):
    # center_wav = np.arange(0, L, L/(0.5*sr)) ** 2
    # center_wav = np.concatenate([center_wav, center_wav[::-1]])
    # center_wav[L*sr//2:3*L*sr//4] = 1
    # only take 0.3 sec audio
    center_wav = np.zeros(int(L * sr))
    center_wav[int(0.4*L*sr):int(0.7*L*sr)] = 1

    return audio * center_wav

def get_spec_librosa(
    wavs,
    sample_rate=16000,
    use_volume_jittering=False,
    center=False,
):
    # Volume  jittering - scale volume by factor in range (0.9, 1.1)
    if use_volume_jittering:
        wavs = [wav * np.random.uniform(0.9, 1.1) for wav in wavs]
    if center:
        wavs = [center_only(wav) for wav in wavs]

    # Convert to log filterbank
    specs = [librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=400,
        hop_length=126,
        n_mels=128,
    ).astype('float32') for wav in wavs]

    # Convert to 32-bit float and expand dim
    specs = [librosa.power_to_db(spec) for spec in specs]
    specs = np.stack(specs, axis=0)
    specs = np.expand_dims(specs, 1)
    specs = torch.as_tensor(specs)  # Nx1xFxT

    return specs


def calcEuclideanDistance_Mat(X, Y):
    """
    Inputs:
    - X: A numpy array of shape (N, F)
    - Y: A numpy array of shape (M, F)

    Returns:
    A numpy array D of shape (N, M) where D[i, j] is the Euclidean distance
    between X[i] and Y[j].
    """
    return ((torch.sum(X ** 2, axis=1, keepdims=True)) + (torch.sum(Y ** 2, axis=1, keepdims=True)).T - 2 * X @ Y.T) ** 0.5


def calcEuclideanDistance(x1, x2):
    return torch.sum((x1 - x2)**2, dim=1)**0.5


def split_data(in_list, portion=(0.9, 0.95), is_shuffle=True):
    if is_shuffle:
        shuffle(in_list)
    if type(in_list) == str:
        with open(in_list) as l:
            fw_list = json.load(l)
    elif type(in_list) == list:
        fw_list = in_list
    else:
        print(type(in_list))
        raise TypeError('Invalid input list type')
    c1, c2 = int(len(fw_list) * portion[0]), int(len(fw_list) * portion[1])
    tr_list, va_list, te_list = fw_list[:c1], fw_list[c1:c2], fw_list[c2:]
    print(
        f'==> train set: {len(tr_list)}, validation set: {len(va_list)}, test set: {len(te_list)}')
    return tr_list, va_list, te_list


def load_one_clip(video_path):
    v = VideoFileClip(video_path)
    fps = int(v.fps)
    frames = [f for f in v.iter_frames()][:-1]
    frame_cnt = len(frames)
    frame_length = 1000./fps
    total_length = int(1000 * (frame_cnt / fps))

    a = v.audio
    sr = a.fps
    a = np.array([fa for fa in a.iter_frames()])
    a = librosa.resample(a, sr, 48000)
    if len(a.shape) > 1:
        a = np.mean(a, axis=1)

    while True:
        idx = np.random.choice(np.arange(frame_cnt - 1), 1)[0]
        frame_clip = frames[idx]
        start_time = int(idx * frame_length + 0.5 * frame_length - 500)
        end_time = start_time + INTERVAL
        if start_time < 0 or end_time > total_length:
            continue
        wave_clip = a[48 * start_time: 48 * end_time]
        if wave_clip.shape[0] != 48000:
            continue
        break
    return frame_clip, wave_clip


def resize_frame(frame):
    H, W = frame.size
    short_edge = min(H, W)
    scale = 256 / short_edge
    H_tar, W_tar = int(np.round(H * scale)), int(np.round(W * scale))
    return frame.resize((H_tar, W_tar))


def get_spectrogram(wave, amp_jitter, amp_jitter_range, log_scale=True, sr=48000):
    # random clip-level amplitude jittering
    if amp_jitter:
        amplified = wave * np.random.uniform(*amp_jitter_range)
        if wave.dtype == np.int16:
            amplified[amplified >= 32767] = 32767
            amplified[amplified <= -32768] = -32768
            wave = amplified.astype('int16')
        elif wave.dtype == np.float32 or wave.dtype == np.float64:
            amplified[amplified >= 1] = 1
            amplified[amplified <= -1] = -1

    # fr, ts, spectrogram = signal.spectrogram(wave[:48000], fs=sr, nperseg=480, noverlap=240, nfft=512)
    # spectrogram = librosa.feature.melspectrogram(S=spectrogram, n_mels=257) # Try log-mel spectrogram?
    spectrogram = librosa.feature.melspectrogram(
        y=wave[:48000], sr=sr, hop_length=240, win_length=480, n_mels=257)
    if log_scale:
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    assert spectrogram.shape[0] == 257

    return spectrogram


def cropAudio(audio, sr, f_idx, fps=10, length=1., left_shift=0):
    time_per_frame = 1./fps
    assert audio.shape[0] > sr * length
    start_time = f_idx * time_per_frame - left_shift
    start_time = 0 if start_time < 0 else start_time
    start_idx = int(np.round(sr * start_time))
    end_idx = int(np.round(start_idx + (sr * length)))
    if end_idx > audio.shape[0]:
        end_idx = audio.shape[0]
        start_idx = int(end_idx - (sr * length))
    try:
        assert audio[start_idx:end_idx].shape[0] == sr * length
    except:
        print(audio.shape, start_idx, end_idx, end_idx - start_idx)
        exit(1)
    return audio[start_idx:end_idx]


def pick_async_frame_idx(idx, total_frames, fps=10, gap=2.0, length=1.0, cnt=1):
    assert idx < total_frames - fps * length
    lower_bound = idx - int((length + gap) * fps)
    upper_bound = idx + int((length + gap) * fps)
    proposal = list(range(0, lower_bound)) + \
        list(range(upper_bound, int(total_frames - fps * length)))
    # assert len(proposal) >= cnt
    avail_cnt = len(proposal)
    try:
        for i in range(cnt - avail_cnt):
            proposal.append(proposal[i % avail_cnt])
    except Exception as e:
        print(idx, total_frames, proposal)
        raise e
    return sample(proposal, k=cnt)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epoch))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
