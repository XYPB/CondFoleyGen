import numpy as np

import torch
import torchaudio.functional
import torchaudio
from . import utils

import pdb


def stft_frame_length(pr): return int(pr.frame_length_ms * pr.samp_sr * 0.001)

def stft_frame_step(pr): return int(pr.frame_step_ms * pr.samp_sr * 0.001)


def stft_num_fft(pr): return int(2**np.ceil(np.log2(stft_frame_length(pr))))

def log10(x): return torch.log(x)/torch.log(torch.tensor(10.))


def db_from_amp(x, cuda=False):
    if cuda: 
        return 20. * log10(torch.max(torch.tensor(1e-5).to('cuda'), x.float()))
    else: 
        return 20. * log10(torch.max(torch.tensor(1e-5), x.float()))


def amp_from_db(x):
    return torch.pow(10., x / 20.)


def norm_range(x, min_val, max_val):
    return 2.*(x - min_val)/float(max_val - min_val) - 1.

def unnorm_range(y, min_val, max_val):
  return 0.5*float(max_val - min_val) * (y + 1) + min_val

def normalize_spec(spec, pr):
    return norm_range(spec, pr.spec_min, pr.spec_max)


def unnormalize_spec(spec, pr):
    return unnorm_range(spec, pr.spec_min, pr.spec_max)


def normalize_phase(phase, pr):
    return norm_range(phase, -np.pi, np.pi)


def unnormalize_phase(phase, pr):
    return unnorm_range(phase, -np.pi, np.pi)


def normalize_ims(im): 
    if type(im) == type(np.array([])): 
        im = im.astype('float32')
    else: 
        im = im.float()
    return -1. + 2. * im


def stft(samples, pr, cuda=False):
    spec_complex = torch.stft(
        samples, 
        stft_num_fft(pr),
        hop_length=stft_frame_step(pr), 
        win_length=stft_frame_length(pr)).transpose(1,2)

    real = spec_complex[..., 0]
    imag = spec_complex[..., 1]
    mag = torch.sqrt((real**2) + (imag**2))
    phase = utils.angle(real, imag)
    if pr.log_spec:
        mag = db_from_amp(mag, cuda=cuda)
    return mag, phase


def make_complex(mag, phase):
    return torch.cat(((mag * torch.cos(phase)).unsqueeze(-1), (mag * torch.sin(phase)).unsqueeze(-1)), -1)


def istft(mag, phase, pr):
    if pr.log_spec:
        mag = amp_from_db(mag)
    # print(make_complex(mag, phase).shape)
    samples = torchaudio.functional.istft(
        make_complex(mag, phase).transpose(1,2),
        stft_num_fft(pr),
        hop_length=stft_frame_step(pr),
        win_length=stft_frame_length(pr))
    return samples



def aud2spec(sample, pr, stereo=False, norm=False, cuda=True): 
    sample = sample[:, :pr.sample_len]
    spec, phase = stft(sample.transpose(1,2).reshape((sample.shape[0]*2, -1)), pr, cuda=cuda)
    spec = spec.reshape(sample.shape[0], 2, pr.spec_len, -1)
    phase = phase.reshape(sample.shape[0], 2, pr.spec_len, -1)
    return spec, phase


def mix_sounds(samples0, pr, samples1=None, cuda=False, dominant=False, noise_ratio=0):
    # pdb.set_trace()
    samples0 = utils.normalize_rms(samples0, pr.input_rms)
    if samples1 is not None:
        samples1 = utils.normalize_rms(samples1, pr.input_rms)

    if dominant: 
        samples0 = samples0[:, :pr.sample_len]
        samples1 = samples1[:, :pr.sample_len] * noise_ratio
    else: 
        samples0 = samples0[:, :pr.sample_len]
        samples1 = samples1[:, :pr.sample_len]
    
    samples_mix = (samples0 + samples1)
    if cuda: 
        samples0 = samples0.to('cuda')
        samples1 = samples1.to('cuda')
        samples_mix = samples_mix.to('cuda')

    spec_mix, phase_mix = stft(samples_mix, pr, cuda=cuda)

    spec0, phase0 = stft(samples0, pr, cuda=cuda)
    spec1, phase1 = stft(samples1, pr, cuda=cuda)

    spec_mix = spec_mix[:, :pr.spec_len]
    phase_mix = phase_mix[:, :pr.spec_len]
    spec0 = spec0[:, :pr.spec_len]
    spec1 = spec1[:, :pr.spec_len]
    phase0 = phase0[:, :pr.spec_len]
    phase1 = phase1[:, :pr.spec_len]

    return utils.Struct(
        samples=samples_mix.float(),
        phase=phase_mix.float(),
        spec=spec_mix.float(),
        sample_parts=[samples0, samples1],
        spec_parts=[spec0.float(), spec1.float()],
        phase_parts=[phase0.float(), phase1.float()])


def pit_loss(pred_spec_fg, pred_spec_bg, snd, pr, cuda=True, vis=False):
    # if pr.norm_spec: 
    def ns(x): return normalize_spec(x, pr)
    # else: 
    #     def ns(x): return x
    if pr.norm:
        gts_ = [[ns(snd.spec_parts[0]), None],
                [ns(snd.spec_parts[1]), None]]
        preds = [[ns(pred_spec_fg), None],
                [ns(pred_spec_bg), None]]
    else:
        gts_ = [[snd.spec_parts[0], None],
                [snd.spec_parts[1], None]]
        preds = [[pred_spec_fg, None],
                [pred_spec_bg, None]]

    def l1(x, y): return torch.mean(torch.abs(x - y), (1, 2))
    losses = []
    for i in range(2):
        gt = [gts_[i % 2], gts_[(i+1) % 2]]
        fg_spec = pr.l1_weight * l1(preds[0][0], gt[0][0])
        bg_spec = pr.l1_weight * l1(preds[1][0], gt[1][0])
        losses.append(fg_spec + bg_spec)

    losses = torch.cat([x.unsqueeze(0) for x in losses], dim=0)
    if vis:
        print(losses)
    loss_val = torch.min(losses, dim=0)
    if vis:
        print(loss_val[1])
    loss = torch.mean(loss_val[0])

    return loss


def diff_loss(spec_diff, phase_diff, snd, pr, device, norm=False, vis=False):
    def ns(x): return normalize_spec(x, pr)
    def np(x): return normalize_phase(x, pr)
    criterion = torch.nn.L1Loss()
    
    gt_spec_diff = snd.spec_diff
    gt_phase_diff = snd.phase_diff
    criterion = criterion.to(device)

    if norm:
        gt_spec_diff = ns(gt_spec_diff)
        gt_phase_diff = np(gt_phase_diff)
        pred_spec_diff = ns(spec_diff)
        pred_phase_diff = np(phase_diff)
    else:
        pred_spec_diff = spec_diff
        pred_phase_diff = phase_diff

    spec_loss = criterion(pred_spec_diff, gt_spec_diff)
    phase_loss = criterion(pred_phase_diff, gt_phase_diff)
    loss = pr.l1_weight * spec_loss + pr.phase_weight * phase_loss
    if vis:
        print(loss)
    return loss

# def pit_loss(out, snd, pr, cuda=False, vis=False):
#     def ns(x): return normalize_spec(x, pr)
#     def np(x): return normalize_phase(x, pr)
#     if cuda: 
#         snd['spec_part0'] = snd['spec_part0'].to('cuda')
#         snd['phase_part0'] = snd['phase_part0'].to('cuda')
#         snd['spec_part1'] = snd['spec_part1'].to('cuda')
#         snd['phase_part1'] = snd['phase_part1'].to('cuda')
#     # gts_ = [[ns(snd['spec_part0'][:, 0, :, :]), np(snd['phase_part0'][:, 0, :, :])],
#     #         [ns(snd['spec_part1'][:, 0, :, :]), np(snd['phase_part1'][:, 0, :, :])]]
#     gts_ = [[ns(snd.spec_parts[0]), np(snd.phase_parts[0])],
#             [ns(snd.spec_parts[1]), np(snd.phase_parts[1])]]
#     preds = [[ns(out.pred_spec_fg), np(out.pred_phase_fg)],
#              [ns(out.pred_spec_bg), np(out.pred_phase_bg)]]
    
#     def l1(x, y): return torch.mean(torch.abs(x - y), (1, 2))
#     losses = []
#     for i in range(2):
#         gt = [gts_[i % 2], gts_[(i+1) % 2]]
#         #   print 'preds[0][0] shape =', shape(preds[0][0])
#         # fg_spec = pr.l1_weight * l1(preds[0][0], gt[0][0])
#         # fg_phase = pr.phase_weight * l1(preds[0][1], gt[0][1])

#         # bg_spec = pr.l1_weight * l1(preds[1][0], gt[1][0])
#         # bg_phase = pr.phase_weight * l1(preds[1][1], gt[1][1])

#         # losses.append(fg_spec + fg_phase + bg_spec + bg_phase)
#         fg_spec = pr.l1_weight * l1(preds[0][0], gt[0][0])

#         bg_spec = pr.l1_weight * l1(preds[1][0], gt[1][0])

#         losses.append(fg_spec + bg_spec)
#     # pdb.set_trace()
#     # pdb.set_trace()
#     losses = torch.cat([x.unsqueeze(0) for x in losses], dim=0)
#     if vis: 
#         print(losses)
#     loss_val = torch.min(losses, dim=0)
#     if vis: 
#         print(loss_val[1])
#     loss = torch.mean(loss_val[0])

#     return loss

# def stereo_mel()


def audio_stft(stft, audio, pr):
    N, C, A = audio.size()
    audio = audio.view(N * C, A)
    spec = stft(audio)
    spec = spec.transpose(-1, -2)
    spec = db_from_amp(spec, cuda=True)
    spec = normalize_spec(spec, pr)
    _, T, F = spec.size()
    spec = spec.view(N, C, T, F)
    return spec


def normalize_audio(samples, desired_rms=0.1, eps=1e-4):
    # print(np.mean(samples**2))
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples