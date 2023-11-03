import os
import time
import json
import sys
from pathlib import Path
from random import sample, shuffle
import argparse

import glob
import soundfile
import torch
import torchvision.transforms as transforms
from specvqgan.data.transforms import *
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from PIL import Image
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
import numpy as np
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models


FRAME_TRANS = transforms.Compose([
            Resize3D(128),
            CenterCrop3D(112),
            ToTensor3D(),
            Normalize3D(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

from feature_extraction.demo_utils import (extract_melspectrogram, load_model,
                                           trim_video, load_frames,
                                           reencode_video_with_diff_fps)
from feature_extraction.demo_utils import (extract_melspectrogram,
                                            get_duration)
from sample_visualization import spec_to_audio_to_st
from specvqgan.modules.losses.vggishish.transforms import Crop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default=None,
                    help='folder name of the pre-trained model')
parser.add_argument('--target_log_dir', type=str, default=None,
                    help='output folder name under logs/ dir')
parser.add_argument('--slide_win_mode', type=str, default='half', choices=['half', 'last'],
                    help='slide window method when generating longer audio')
parser.add_argument('--W_scale', type=int, default=1,
                    help='length scale of the generate audio, output will be W_scale*2s')
parser.add_argument('--max_W_scale', type=int, default=3,
                    help='maximum W_scale to iterate')
parser.add_argument('--min_W_scale', type=int, default=1,
                    help='minimum W_scale to iterate')
parser.add_argument('--gen_cnt', type=int, default=30,
                    help='generation count when generating multiple result')
parser.add_argument('--spec_take_first', type=int, default=160,
                    help='cut the spectrogram to this size')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature of softmax for logits to probability')
parser.add_argument('--split', type=int, default=-1,
                    help='split idx when running multi-process generation')
parser.add_argument('--total_split', type=int, default=1,
                    help='total number of multi-process')
parser.add_argument('--tmp_idx', type=int, default=-1,
                    help='temperate folder idx to place intermediate result')
parser.add_argument('--new_codebook', action='store_true',
                    help='load a different codebook according to config')
parser.add_argument('--gh_testset', action='store_true',
                    help='running greatest hit testset')
parser.add_argument('--gh_demo', action='store_true',
                    help='running the greatest hit demo')
parser.add_argument('--gh_gen', action='store_true',
                    help='generate audio with greatest hit model')
parser.add_argument('--countix_av_gen', action='store_true',
                    help='generate audio with countix-AV model')
parser.add_argument('--multiple', action='store_true',
                    help='generate multiple audio for each pair of input for re-ranking')
parser.add_argument('--style_transfer', action='store_true',
                    help='add to set this option to true')

class CropImage(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def load_specs_as_img(spec, spec_take_first=192):
    loader = transforms.Compose([
            transforms.Resize((80, spec_take_first)),  # scale imported image
            transforms.ToTensor()])  # transform it into a torch tensor
    spec = spec[:, :spec_take_first]
    spec_img = Image.fromarray((spec * 255).astype(np.uint8)).convert('RGB')
    spec_img = loader(spec_img).unsqueeze(0)
    return spec_img.to(device, torch.float)

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 150 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def attach_audio_to_video(video_path, audio_path, dest, start_step, FPS=15, recon_only=False, put_text=False, v_duration=2):
    clip = VideoFileClip(video_path).set_fps(FPS)
    if put_text:
        frames = [f for f in clip.iter_frames()]
        H, W, _ = frames[0].shape
        for i in range(len(frames)):
            text = 'Original Audio' if i < start_step else 'Generated Audio'
            if recon_only:
                text = 'Reconstructed Sound'
            img_w_text = cv2.putText(frames[i], text, (W//50, H//6), 
                                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                    color=(255, 0, 0), thickness=3)
        clip = ImageSequenceClip(frames, fps=FPS)
    clip = clip.subclip(0, v_duration)
    clip = clip.set_audio(AudioFileClip(audio_path))
    clip.write_videofile(dest, audio=True, fps=FPS, verbose=False, logger=None)
    return clip


def draw_spec(spec, dest, cmap='magma'):
    plt.imshow(spec, cmap=cmap, origin='lower')
    plt.axis('off')
    plt.savefig(dest, bbox_inches='tight', pad_inches=0.)
    plt.close()




def gen_audio_condImage_fast(video_path,
                            extra_cond_video_path,
                            model,
                            target_log_dir = 'CondAVTransformer',
                            cond_cnt=0,
                            SR = 22050,
                            FPS = 15,
                            L = 2.0,
                            normalize=False,
                            using_torch=False,
                            show_griffin_lim=False,
                            vqgan_L=10.0,
                            style_transfer=False,
                            target_start_time=0,
                            cond_start_time=0,
                            outside=False,
                            remove_noise=False,
                            spec_take_first=160,
                            W_scale=1,
                            slide_win_mode='half',
                            temperature = 1.0,
                            ignore_input_spec=False,
                            tmp_path='./tmp',
                            fix_cond=True):
    '''
    parameters:
        video_path: path to the target video, will be trimmed to 2s and re-encode into 15 fps. 
        extra_cond_video_path: path to the conditional video, will be trimmed to 2s and re-encode into 15 fps. 
        model: model object, returned by load_model function
        target_log_dir: target output dir name in the 'logs' directory, e.g. output will be saved to 'logs/<target_log_dir>'
        cond_cnt: index of current condition video
        SR: sampling rate
        FPS: Frame rate
        L: length of generated sound
        normalize: whether to normaliza input waveform
        using_torch: use torchaudio to extrac spectrogram
        show_griffin_lim: use griffin_lim algorithm vocoder
        vqgan_L: length of VQ-GAN codebook, use 2 if using GreatestHit codebook
        style_transfer: generate style transfer sound
        target_start_time: if target video is from outside, trim from <target_start_time> to <target_start_time>+2
        cond_start_time: if conditional video is from outside, trim from <cond_start_time> to <cond_start_time>+2
        outside: indicate whether the video from outside source
        remove_noise: denoise for outside videos
        spec_take_first: size of the spectrogram to use
        W_scale: scale of audio duration as multiples of 2sec
        slide_win_mode: mode of sliding window, choose from ['half', 'last']
        temperature: temperature of multinomial sampling.
        ignore_input_spec: ignore input spec when input video is silent
        tmp_path: tmp dir to save intermediate files
        fix_cond: use only 2 sec condition regardless to input length.
    '''

    config, sampler, melgan, melception = model
    # feature extractor
    L = int(L * W_scale)
    vqgan_L = int(vqgan_L * W_scale)
    spec_take_first = int(spec_take_first * W_scale)
    if '_denoised_' not in video_path or outside:
        new_fps_video_path = reencode_video_with_diff_fps(video_path, tmp_path, FPS)
        video_path = trim_video(new_fps_video_path, target_start_time, vqgan_L, tmp_path=tmp_path)
    frames = [Image.fromarray(f) for f in load_frames(video_path)][:int(FPS * L)]
    frames = FRAME_TRANS(frames)

    if '_denoised_' not in extra_cond_video_path or outside:
        new_fps_video_path = reencode_video_with_diff_fps(extra_cond_video_path, tmp_path, FPS)
        extra_cond_video_path = trim_video(new_fps_video_path, cond_start_time, vqgan_L, tmp_path=tmp_path)
    cond_frames = [Image.fromarray(f) for f in load_frames(extra_cond_video_path)][:int(FPS * L)]
    cond_frames = FRAME_TRANS(cond_frames)

    feats = {'feature': np.stack(cond_frames + frames, axis=0)}

    cond_video_path = extra_cond_video_path
    tar_video_path = video_path

    # Extract Features
    visual_features = feats

    # Prepare Input
    batch = default_collate([visual_features])
    batch['feature'] = batch['feature'].to(device)
    with torch.no_grad():
        c = sampler.get_input(sampler.cond_stage_key, batch)

    # Extract Spectrogram
    if not ignore_input_spec:
        spectrogram = extract_melspectrogram(tar_video_path, SR, normalize=normalize, using_torch=using_torch, remove_noise=remove_noise, duration=vqgan_L, tmp_path=tmp_path)
        spec_H, spec_W = spectrogram.shape
        if spec_W > spec_take_first:
            spectrogram = spectrogram[:, :spec_take_first]
        else:
            pad = np.zeros((spec_H, spec_take_first), dtype=spectrogram.dtype)
            pad[:, :spec_W] = spectrogram
            spectrogram = pad
        spectrogram = {'input': spectrogram}
        if config.data.params.spec_crop_len is None or W_scale != 1:
            config.data.params.spec_crop_len = spec_take_first
        if spectrogram['input'].shape[1] > config.data.params.spec_crop_len:
            random_crop = False
            crop_img_fn = CropImage([config.data.params.mel_num, config.data.params.spec_crop_len], random_crop)
            spectrogram = crop_img_fn(spectrogram)
        # Prepare input
        batch = default_collate([spectrogram])
        batch['image'] = batch['input'].to(device)
        x = sampler.get_input(sampler.first_stage_key, batch)
        mel_x = x.detach().cpu().numpy()

        # Encode and Decode the Spectrogram
        with torch.no_grad():
            quant_z, z_indices = sampler.encode_to_z(x)
            # print(z_indices)
            xrec = sampler.first_stage_model.decode(quant_z)
            mel_xrec = xrec.detach().cpu().numpy()
    
    # Conditional
    # Extract Spectrogram
    spectrogram = extract_melspectrogram(cond_video_path, SR, normalize=normalize, using_torch=using_torch, remove_noise=remove_noise, duration=vqgan_L, tmp_path=tmp_path)
    spec_H, spec_W = spectrogram.shape
    if spec_W > spec_take_first:
        padded = False
        spectrogram = spectrogram[:, :spec_take_first]
    else:
        padded = True
        pad = np.zeros((spec_H, spec_take_first), dtype=spectrogram.dtype)
        pad[:, :spec_W] = spectrogram
        orig_width = spec_W
        spectrogram = pad
    spectrogram = {'input': spectrogram}
    if config.data.params.spec_crop_len is None or W_scale != 1:
        config.data.params.spec_crop_len = spec_take_first
    if spectrogram['input'].shape[1] > config.data.params.spec_crop_len:
        random_crop = False
        crop_img_fn = CropImage([config.data.params.mel_num, config.data.params.spec_crop_len], random_crop)
        spectrogram = crop_img_fn(spectrogram)

    # Prepare input
    batch = default_collate([spectrogram])
    batch['cond_image'] = batch['input'].to(device)
    xp = sampler.get_input(sampler.cond_first_stage_key, batch)
    mel_xp = xp.detach().cpu().numpy()

    # Encode and Decode the Spectrogram
    with torch.no_grad():
        quant_zp, zp_indices = sampler.encode_to_z(xp)
        # print(zp_indices)
        xprec = sampler.first_stage_model.decode(quant_zp)
        mel_xprec = xprec.detach().cpu().numpy()

    if ignore_input_spec:
        z_indices = torch.zeros_like(zp_indices)
        xrec = torch.zeros_like(xprec)
        mel_xrec = np.zeros_like(mel_xprec)

    # Define Sampling Parameters
    # take top 1024 / 512 code
    top_x = sampler.first_stage_model.quantize.n_e // 2
    
    if not os.path.exists(f'logs/{target_log_dir}'):
        os.mkdir(f'logs/{target_log_dir}')
    
    target_dir = os.path.join(f'logs/{target_log_dir}', f'2sec_full_generated_sound_{cond_cnt}')
    target_v_dir = os.path.join(f'logs/{target_log_dir}', f'2sec_full_generated_video_{cond_cnt}')
    target_cond_v_dir = os.path.join(f'logs/{target_log_dir}', f'2sec_full_cond_video_{cond_cnt}')
    target_orig_v_dir = os.path.join(f'logs/{target_log_dir}', f'2sec_full_orig_video')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    if not os.path.exists(target_v_dir):
        os.mkdir(target_v_dir)
    if not os.path.exists(target_cond_v_dir):
        os.mkdir(target_cond_v_dir)
    if not os.path.exists(target_orig_v_dir):
        os.mkdir(target_orig_v_dir)
    # Start sampling
    if style_transfer:
        content_img = load_specs_as_img(mel_xrec[0, 0, :, :spec_take_first])
        style_img = load_specs_as_img(mel_xprec[0, 0, :, :spec_take_first])
        generated_spec = run_style_transfer(
            cnn_normalization_mean.to(),
            cnn_normalization_std.to(),
            content_img.clone().to(device),
            style_img.clone().to(device),
            content_img.clone().to(device),
        )
        z_pred_img = torch.mean(generated_spec, dim=1, keepdim=True)
        mel_z = z_pred_img.detach().cpu().numpy()
    else:
        with torch.no_grad():
            start_t = time.time()

            quant_c, c_indices = sampler.encode_to_c(c)
            z_indices_clip = z_indices[:, :sampler.clip * W_scale]
            zp_indices_clip = zp_indices[:, :sampler.clip * W_scale]
            z_indices_rec = z_indices.clone()
            # crec = sampler.cond_stage_model.decode(quant_c)

            patch_size_i = 5

            c_window_size = int(2 * FPS)

            downsampled_size = spec_take_first // 16
            cond_patch_shift_j = (W_scale - 1) * (downsampled_size // W_scale)
            if 'dropcond_' in target_log_dir:
                B, D, hr_h, hr_w = sampling_shape = (1, 256, 5, int(downsampled_size))
                patch_size_j = int(downsampled_size // W_scale)
            else:
                B, D, hr_h, hr_w = sampling_shape = (1, 256, 5, int(2*downsampled_size))
                patch_size_j = int(2*downsampled_size // W_scale)
            z_pred_indices = torch.zeros((B, hr_h*hr_w)).long().to(device)

            if 'dropcond_' not in target_log_dir:
                start_step = zp_indices_clip.shape[1]
                z_pred_indices[:, :start_step] = zp_indices_clip[:, :start_step]
            elif 'dropcond_' in target_log_dir:
                start_step = 0

            pbar = tqdm(range(start_step, hr_w * hr_h), desc='Sampling Codebook Indices')
            for step in pbar:
                i = step % hr_h
                j = step // hr_h

                i_start = min(max(0, i - (patch_size_i // 2)), hr_h - patch_size_i)
                # only last
                # 
                if slide_win_mode == 'half':
                    j_start = min(max(0, j - (3 * patch_size_j // 4)), hr_w - patch_size_j)
                elif slide_win_mode == 'last':
                    j_start = min(max(0, j -  patch_size_j + 1), hr_w - patch_size_j)
                else:
                    raise NotImplementedError
                i_end = i_start + patch_size_i
                j_end = j_start + patch_size_j

                local_i = i - i_start

                patch_2d_shape = (B, D, patch_size_i, patch_size_j)

                if W_scale != 1:
                    cond_j_start = 0 if fix_cond else max(0, j_start - cond_patch_shift_j)
                    cond_j_end = cond_j_start + (patch_size_j // 2)
                    tar_j_start = max(0, j_start - cond_patch_shift_j) + (hr_w // 2)
                    tar_j_end = tar_j_start + (patch_size_j // 2)

                    local_j = j - tar_j_start + (patch_size_j // 2)

                    pbar.set_postfix(
                        Step=f'({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end}, {cond_j_start}:{cond_j_end}|{tar_j_start}:{tar_j_end})'
                    )
                    cond_patch = z_pred_indices \
                        .reshape(B, hr_w, hr_h) \
                        .permute(0, 2, 1)[:, i_start:i_end, cond_j_start:cond_j_end].permute(0, 2, 1)
                    tar_patch = z_pred_indices \
                        .reshape(B, hr_w, hr_h) \
                        .permute(0, 2, 1)[:, i_start:i_end, tar_j_start:tar_j_end].permute(0, 2, 1)
                    patch = torch.cat([cond_patch, tar_patch], dim=1).reshape(B, patch_size_i * patch_size_j)

                    cond_t_start = cond_j_start * 0.2
                    cond_frame_start = int(cond_t_start * FPS)
                    cond_frame_end = cond_frame_start + c_window_size
                    tar_frame_start = int(cond_frame_start + c_window_size * W_scale)
                    tar_frame_end = tar_frame_start + c_window_size
                    cpatch = torch.cat([c_indices[:, :, cond_frame_start:cond_frame_end], c_indices[:, :, tar_frame_start:tar_frame_end]], dim=2)
                else:
                    local_j = j - j_start
                    pbar.set_postfix(
                        Step=f'({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})'
                    )
                    patch = z_pred_indices
                    # assuming we don't crop the conditioning and just use the whole c, if not desired uncomment the above
                    cpatch = c_indices

                logits, _, attention = sampler.transformer(patch[:, :-1], cpatch)
                # remove conditioning
                logits = logits[:, -patch_size_j*patch_size_i:, :]

                local_pos_in_flat = local_j * patch_size_i + local_i
                logits = logits[:, local_pos_in_flat, :]

                logits = logits / temperature
                logits = sampler.top_k_logits(logits, top_x)

                # apply softmax to convert to probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)

                # sample from the distribution
                ix = torch.multinomial(probs, num_samples=1)
                z_pred_indices[:, j * hr_h + i] = ix

            # quant_z_shape = sampling_shape
            if 'dropcond_' in target_log_dir:
                z_indices_rec[:, :sampler.clip * W_scale] = z_pred_indices
            else:
                z_indices_rec[:, :sampler.clip * W_scale] = z_pred_indices[:, sampler.clip * W_scale:]
            # print(z_indices_rec)
            z_pred_img = sampler.decode_to_img(z_indices_rec, 
                                            (1, 256, 5, 53 if vqgan_L == 10.0 else downsampled_size))
            mel_z = z_pred_img.detach().cpu().numpy()

    with torch.no_grad():
        config.data.params.spec_dir_path = 'melspec_10s_22050hz'
        if padded:
            z_pred_img = z_pred_img[:, :, :, :orig_width]
            xrec = xrec[:, :, :, :orig_width]
            xprec = xprec[:, :, :, :orig_width]
            spec_take_first = orig_width
        waves = spec_to_audio_to_st(z_pred_img, config.data.params.spec_dir_path,
                                    config.data.params.sample_rate, show_griffin_lim=show_griffin_lim,
                                    vocoder=melgan, show_in_st=False)
        
        # Original Reconstruction
        orig_waves = spec_to_audio_to_st(xrec, config.data.params.spec_dir_path,
                                    config.data.params.sample_rate, show_griffin_lim=show_griffin_lim,
                                    vocoder=melgan, show_in_st=False)

        # Conditional Reconstruction
        cond_waves = spec_to_audio_to_st(xprec, config.data.params.spec_dir_path,
                                    config.data.params.sample_rate, show_griffin_lim=show_griffin_lim,
                                    vocoder=melgan, show_in_st=False)


    if show_griffin_lim:
        waves['vocoder'] = waves['inv_transforms'][:int(22050 * L)]
    else:
        waves['vocoder'] = waves['vocoder'][:int(22050 * L)]
    _cond_video_path = cond_video_path
    save_path = os.path.join(target_dir, Path(tar_video_path).stem +
                             '_to_' + Path(_cond_video_path).stem + '.wav')
    soundfile.write(save_path, waves['vocoder'], config.data.params.sample_rate, 'PCM_24')
    print(f'The sample has been saved @ {save_path}')
    save_video_path = os.path.join(target_v_dir, Path(tar_video_path).stem +
                                   '_to_' + Path(_cond_video_path).stem + '.mp4')
    attach_audio_to_video(tar_video_path, save_path, save_video_path, 0, v_duration=L)
    
    # Original sound attach
    if show_griffin_lim:
        waves['vocoder'] = waves['inv_transforms'][:int(22050 * L)]
    else:
        waves['vocoder'] = waves['vocoder'][:int(22050 * L)]
    orig_save_path = os.path.join(target_orig_v_dir, Path(tar_video_path).stem + '.wav')
    soundfile.write(orig_save_path, orig_waves['vocoder'], config.data.params.sample_rate, 'PCM_24')
    print(f'The sample has been saved @ {orig_save_path}')
    orig_save_video_path = os.path.join(target_orig_v_dir, Path(tar_video_path).stem + '.mp4')
    attach_audio_to_video(tar_video_path, orig_save_path, orig_save_video_path, 0, recon_only=True, v_duration=L)
    
    # Conditional sound attach
    _cond_video_path = cond_video_path
    # Save only the first 2sec conditional audio+video if fix_cond
    _L = L // W_scale if fix_cond else L
    if show_griffin_lim:
        waves['vocoder'] = waves['inv_transforms'][:int(22050 * _L)]
    else:
        waves['vocoder'] = waves['vocoder'][:int(22050 * _L)]
    cond_save_path = os.path.join(target_cond_v_dir, Path(tar_video_path).stem +
                                  '_to_' + Path(_cond_video_path).stem + '.wav')
    soundfile.write(cond_save_path, cond_waves['vocoder'], config.data.params.sample_rate, 'PCM_24')
    print(f'The sample has been saved @ {cond_save_path}')
    cond_save_video_path = os.path.join(target_cond_v_dir, Path(tar_video_path).stem +
                                        '_to_' + Path(_cond_video_path).stem + '.mp4')
    attach_audio_to_video(_cond_video_path, cond_save_path, cond_save_video_path, 0, recon_only=True, v_duration=_L)
    
    
    # plot melspec
    # target
    plt.imshow(mel_xrec[0, 0, :, :spec_take_first], cmap='coolwarm', origin='lower')
    plt.axis('off')
    plt.savefig(orig_save_video_path.replace('.mp4', '.jpg'), bbox_inches='tight', pad_inches=0.)
    plt.close()
    
    # condition
    _spec_take_first = int(spec_take_first // W_scale) if fix_cond else spec_take_first
    plt.imshow(mel_xprec[0, 0, :, :_spec_take_first], cmap='coolwarm', origin='lower')
    plt.axis('off')
    plt.savefig(cond_save_video_path.replace('.mp4', '.jpg'), bbox_inches='tight', pad_inches=0.)
    plt.close()

    # generated
    draw_spec(mel_z[0, 0, :, :spec_take_first], save_video_path.replace('.mp4', '.jpg'), cmap='coolwarm')
    return



def gen_audio_condImage_fast_multiple(video_path,
                            extra_cond_video_path,
                            model,
                            all_gen_dict,
                            target_log_dir = 'CondAVTransformer',
                            SR = 22050,
                            FPS = 15,
                            L = 2.0,
                            normalize=False,
                            using_torch=False,
                            show_griffin_lim=False,
                            vqgan_L=10.0,
                            style_transfer=False,
                            target_start_time=0,
                            cond_start_time=0,
                            outside=False,
                            remove_noise=False,
                            spec_take_first=160,
                            gen_cnt=25,
                            W_scale=1,
                            slide_win_mode='half',
                            temperature=1.0,
                            ignore_input_spec=False,
                            tmp_path='./tmp',
                            fix_cond=True):
    '''
    parameters:
        video_path: path to the target video, will be trimmed to 2s and re-encode into 15 fps. 
        extra_cond_video_path: path to the conditional video, will be trimmed to 2s and re-encode into 15 fps. 
        model: model object, returned by load_model function
        target_log_dir: target output dir name in the 'logs' directory, e.g. output will be saved to 'logs/<target_log_dir>'
        SR: sampling rate
        FPS: Frame rate
        L: length of generated sound
        normalize: whether to normaliza input waveform
        using_torch: use torchaudio to extrac spectrogram
        show_griffin_lim: use griffin_lim algorithm vocoder
        vqgan_L: length of VQ-GAN codebook, use 2 if using GreatestHit codebook
        style_transfer: generate style transfer sound
        target_start_time: if target video is from outside, trim from <target_start_time> to <target_start_time>+2
        cond_start_time: if conditional video is from outside, trim from <cond_start_time> to <cond_start_time>+2
        outside: indicate whether the video from outside source
        remove_noise: denoise for outside videos
        spec_take_first: size of the spectrogram to use
        gen_cnt: count of generation times
        W_scale: scale of audio duration as multiples of 2sec
        slide_win_mode: mode of sliding window, choose from ['half', 'last']
        temperature: temperature of multinomial sampling.
        ignore_input_spec: ignore input spec when input video is silent
        tmp_path: tmp dir to save intermediate files
        fix_cond: use only 2 sec condition regardless to input length.
    '''

    config, sampler, melgan, melception = model
    # feature extractor
    L = int(L * W_scale)
    vqgan_L = int(vqgan_L * W_scale)
    spec_take_first = int(spec_take_first * W_scale)
    if '_denoised_' not in video_path or outside:
        new_fps_video_path = reencode_video_with_diff_fps(video_path, tmp_path, FPS)
        video_path = trim_video(new_fps_video_path, target_start_time, vqgan_L, tmp_path=tmp_path)
    frames = [Image.fromarray(f) for f in load_frames(video_path)][:int(FPS * L)]
    frames = FRAME_TRANS(frames)

    if '_denoised_' not in extra_cond_video_path or outside:
        new_fps_video_path = reencode_video_with_diff_fps(extra_cond_video_path, tmp_path, FPS)
        extra_cond_video_path = trim_video(new_fps_video_path, cond_start_time, vqgan_L, tmp_path=tmp_path)
    cond_frames = [Image.fromarray(f) for f in load_frames(extra_cond_video_path)][:int(FPS * L)]
    cond_frames = FRAME_TRANS(cond_frames)

    feats = {'feature': np.stack(cond_frames + frames, axis=0)}

    cond_video_path = extra_cond_video_path
    tar_video_path = video_path

    # Extract Features
    visual_features = feats

    # Prepare Input
    batch = default_collate([visual_features])
    batch['feature'] = batch['feature'].to(device)
    with torch.no_grad():
        c = sampler.get_input(sampler.cond_stage_key, batch)

    if not ignore_input_spec:
        # Extract Spectrogram
        spectrogram = extract_melspectrogram(tar_video_path, SR, normalize=normalize, using_torch=using_torch, remove_noise=remove_noise, duration=vqgan_L, tmp_path=tmp_path)
        spec_H, spec_W = spectrogram.shape
        if spec_W > spec_take_first:
            spectrogram = spectrogram[:, :spec_take_first]
        else:
            pad = np.zeros((spec_H, spec_take_first), dtype=spectrogram.dtype)
            pad[:, :spec_W] = spectrogram
            spectrogram = pad
        spectrogram = {'input': spectrogram}
        if config.data.params.spec_crop_len is None or W_scale != 1:
            config.data.params.spec_crop_len = spec_take_first
        if spectrogram['input'].shape[1] > config.data.params.spec_crop_len:
            random_crop = False
            crop_img_fn = CropImage([config.data.params.mel_num, config.data.params.spec_crop_len], random_crop)
            spectrogram = crop_img_fn(spectrogram)

        # Prepare input
        batch = default_collate([spectrogram])
        batch['image'] = batch['input'].to(device)
        x = sampler.get_input(sampler.first_stage_key, batch)
        mel_x = x.detach().cpu().numpy()

        # Encode and Decode the Spectrogram
        with torch.no_grad():
            quant_z, z_indices = sampler.encode_to_z(x)
            # print(z_indices)
            xrec = sampler.first_stage_model.decode(quant_z)
            mel_xrec = xrec.detach().cpu().numpy()
    
    # Conditional
    # Extract Spectrogram
    spectrogram = extract_melspectrogram(cond_video_path, SR, normalize=normalize, using_torch=using_torch, remove_noise=remove_noise, duration=vqgan_L, tmp_path=tmp_path)
    spec_H, spec_W = spectrogram.shape
    if spec_W > spec_take_first:
        spectrogram = spectrogram[:, :spec_take_first]
    else:
        pad = np.zeros((spec_H, spec_take_first), dtype=spectrogram.dtype)
        pad[:, :spec_W] = spectrogram
        spectrogram = pad
    spectrogram = {'input': spectrogram}
    if config.data.params.spec_crop_len is None or W_scale != 1:
        config.data.params.spec_crop_len = spec_take_first
    if spectrogram['input'].shape[1] > config.data.params.spec_crop_len:
        random_crop = False
        crop_img_fn = CropImage([config.data.params.mel_num, config.data.params.spec_crop_len], random_crop)
        spectrogram = crop_img_fn(spectrogram)

    # Prepare input
    batch = default_collate([spectrogram])
    batch['cond_image'] = batch['input'].to(device)
    xp = sampler.get_input(sampler.cond_first_stage_key, batch)
    mel_xp = xp.detach().cpu().numpy()

    # Encode and Decode the Spectrogram
    with torch.no_grad():
        quant_zp, zp_indices = sampler.encode_to_z(xp)
        # print(zp_indices)
        xprec = sampler.first_stage_model.decode(quant_zp)
        mel_xprec = xprec.detach().cpu().numpy()

    if ignore_input_spec:
        z_indices = torch.zeros_like(zp_indices)
        xrec = torch.zeros_like(xprec)
        mel_xrec = np.zeros_like(mel_xprec)

    # Define Sampling Parameters
    # take top 1024 / 512 code
    top_x = sampler.first_stage_model.quantize.n_e // 2
    
    if not os.path.exists(f'logs/{target_log_dir}'):
        os.mkdir(f'logs/{target_log_dir}')
    
    if video_path not in all_gen_dict.keys():
        all_gen_dict[video_path] = {}
    all_gen_dict[video_path][extra_cond_video_path] = []
    # Start sampling
    if style_transfer:
        content_img = load_specs_as_img(mel_xrec[0, 0, :, :spec_take_first])
        style_img = load_specs_as_img(mel_xprec[0, 0, :, :spec_take_first])
        generated_spec = run_style_transfer(
            cnn_normalization_mean.to(),
            cnn_normalization_std.to(),
            content_img.clone().to(device),
            style_img.clone().to(device),
            content_img.clone().to(device),
        )
        z_pred_img = torch.mean(generated_spec, dim=1, keepdim=True)
        mel_z = z_pred_img.detach().cpu().numpy()
    else:
        for _ in range(gen_cnt):
            with torch.no_grad():
                start_t = time.time()

                quant_c, c_indices = sampler.encode_to_c(c)
                z_indices_clip = z_indices[:, :sampler.clip * W_scale]
                zp_indices_clip = zp_indices[:, :sampler.clip * W_scale]
                z_indices_rec = z_indices.clone()
                # crec = sampler.cond_stage_model.decode(quant_c)

                patch_size_i = 5

                c_window_size = int(2 * FPS)

                #TODO: modify the shape if drop condition info
                downsampled_size = spec_take_first // 16
                cond_patch_shift_j = (W_scale - 1) * (downsampled_size // W_scale)
                if 'dropcond_' in target_log_dir:
                    B, D, hr_h, hr_w = sampling_shape = (1, 256, 5, int(downsampled_size))
                    patch_size_j = int(downsampled_size // W_scale)
                else:
                    B, D, hr_h, hr_w = sampling_shape = (1, 256, 5, int(2*downsampled_size))
                    patch_size_j = int(2*downsampled_size // W_scale)
                z_pred_indices = torch.zeros((B, hr_h*hr_w)).long().to(device)

                if 'dropcond_' not in target_log_dir:
                    start_step = zp_indices_clip.shape[1]
                    z_pred_indices[:, :start_step] = zp_indices_clip[:, :start_step]
                elif 'dropcond_' in target_log_dir:
                    start_step = 0

                for step in range(start_step, hr_w * hr_h):
                    i = step % hr_h
                    j = step // hr_h

                    i_start = min(max(0, i - (patch_size_i // 2)), hr_h - patch_size_i)
                    if slide_win_mode == 'half':
                        j_start = min(max(0, j - (3 * patch_size_j // 4)), hr_w - patch_size_j)
                    elif slide_win_mode == 'last':
                        j_start = min(max(0, j -  patch_size_j + 1), hr_w - patch_size_j)
                    else:
                        raise NotImplementedError
                    i_end = i_start + patch_size_i
                    j_end = j_start + patch_size_j

                    local_i = i - i_start

                    patch_2d_shape = (B, D, patch_size_i, patch_size_j)

                    if W_scale != 1:
                        # if fix cond, we always use first 2 sec of cond audio.
                        cond_j_start = 0 if fix_cond else max(0, j_start - cond_patch_shift_j)
                        cond_j_end = cond_j_start + (patch_size_j // 2)
                        tar_j_start = max(0, j_start - cond_patch_shift_j) + (hr_w // 2)
                        tar_j_end = tar_j_start + (patch_size_j // 2)

                        local_j = j - tar_j_start + (patch_size_j // 2)

                        cond_patch = z_pred_indices \
                            .reshape(B, hr_w, hr_h) \
                            .permute(0, 2, 1)[:, i_start:i_end, cond_j_start:cond_j_end].permute(0, 2, 1)
                        tar_patch = z_pred_indices \
                            .reshape(B, hr_w, hr_h) \
                            .permute(0, 2, 1)[:, i_start:i_end, tar_j_start:tar_j_end].permute(0, 2, 1)
                        patch = torch.cat([cond_patch, tar_patch], dim=1).reshape(B, patch_size_i * patch_size_j)

                        cond_t_start = cond_j_start * 0.2
                        cond_frame_start = int(cond_t_start * FPS)
                        cond_frame_end = cond_frame_start + c_window_size
                        tar_frame_start = int(cond_frame_start + c_window_size * W_scale)
                        tar_frame_end = tar_frame_start + c_window_size
                        cpatch = torch.cat([c_indices[:, :, cond_frame_start:cond_frame_end], c_indices[:, :, tar_frame_start:tar_frame_end]], dim=2)
                    else:
                        local_j = j - j_start
                        patch = z_pred_indices
                        # assuming we don't crop the conditioning and just use the whole c, if not desired uncomment the above
                        cpatch = c_indices

                    logits, _, attention = sampler.transformer(patch[:, :-1], cpatch)
                    # remove conditioning
                    logits = logits[:, -patch_size_j*patch_size_i:, :]

                    local_pos_in_flat = local_j * patch_size_i + local_i
                    logits = logits[:, local_pos_in_flat, :]

                    logits = logits / temperature
                    logits = sampler.top_k_logits(logits, top_x)

                    # apply softmax to convert to probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)

                    # sample from the distribution
                    ix = torch.multinomial(probs, num_samples=1)
                    z_pred_indices[:, j * hr_h + i] = ix

                # quant_z_shape = sampling_shape
                if 'dropcond_' in target_log_dir:
                    z_indices_rec[:, :sampler.clip * W_scale] = z_pred_indices
                else:
                    z_indices_rec[:, :sampler.clip * W_scale] = z_pred_indices[:, sampler.clip * W_scale:]
                # print(z_indices_rec)
                z_pred_img = sampler.decode_to_img(z_indices_rec, 
                                                (1, 256, 5, 53 if vqgan_L == 10.0 else downsampled_size))
                mel_z = z_pred_img.detach().cpu().numpy()
                config.data.params.spec_dir_path = 'melspec_10s_22050hz'
                waves = spec_to_audio_to_st(z_pred_img, config.data.params.spec_dir_path,
                                            config.data.params.sample_rate, show_griffin_lim=show_griffin_lim,
                                            vocoder=melgan, show_in_st=False)
                waves['vocoder'] = waves['vocoder'][:int(22050 * L)]
                all_gen_dict[video_path][extra_cond_video_path].append(waves['vocoder'])

    return


if __name__ == '__main__':
    args = parser.parse_args()

    model_name = args.model_name
    log_dir = './logs'
    model = load_model(model_name, log_dir, device, args.new_codebook)
    target_log_dir = args.target_log_dir
    if args.tmp_idx == -1:
        tmp_path = f'./tmp/tmp_{args.split}'
    else:
        tmp_path = f'./tmp/tmp_{args.tmp_idx}'
    os.makedirs(tmp_path, exist_ok=True)
    os.makedirs(f'logs/{target_log_dir}', exist_ok=True)
    random.seed(3704)

    slide_win_mode = args.slide_win_mode

    style_transfer = args.style_transfer
    if args.gh_demo:
        orig_videos = ['data/demo_video/hitting_metal.mp4']
        cond_videos = ['data/demo_video/hitting_bag.mp4']
        for i, v in enumerate(orig_videos):
                video_path = v
                start_sec = None
                for j, ep in enumerate(cond_videos):
                    gen_audio_condImage_fast(video_path, ep, model, spec_take_first=args.spec_take_first,
                                            target_log_dir=target_log_dir, using_torch=True,
                                            L=2.0, cond_cnt=j, style_transfer=style_transfer,
                                            normalize=False, show_griffin_lim=False, vqgan_L=2.0)
    elif args.gh_testset:
        fixed_test = json.load(open('./data/AMT_test_set_path.json', 'r'))
        if args.multiple:
            # generate 100 for single bad example.
            split = args.split
            num_splits = args.total_split
            all_gen_dict = {}
            gen_cnt = 100
            dest = os.path.join(f'logs/{target_log_dir}', f'{gen_cnt}_times_split_{split}_wav_dict.pt')
            for i, (v, extra_video_paths) in enumerate(tqdm(fixed_test.items())):
                video_path = v
                start_sec = None
                if i % num_splits != split:
                    continue
                for j, ep in enumerate(extra_video_paths):
                    gen_audio_condImage_fast_multiple(video_path, ep, model, all_gen_dict,
                                                    spec_take_first=args.spec_take_first, gen_cnt=gen_cnt,
                                                    target_log_dir=target_log_dir, using_torch=True,
                                                    L=2.0, cond_cnt=j, style_transfer=style_transfer,
                                                    normalize=False, show_griffin_lim=False, vqgan_L=2.0)
            torch.save(all_gen_dict, dest)
        else:
            for i, (v, extra_video_paths) in enumerate(fixed_test.items()):
                video_path = v
                start_sec = None
                for j, ep in enumerate(extra_video_paths):
                    gen_audio_condImage_fast(video_path, ep, model, spec_take_first=args.spec_take_first,
                                            target_log_dir=target_log_dir, using_torch=True,
                                            L=2.0, cond_cnt=j, style_transfer=style_transfer,
                                            normalize=False, show_griffin_lim=False, vqgan_L=2.0,
                                            W_scale=args.W_scale)
    elif args.gh_gen:
        orig_videos = ['data/demo_video/hitting_metal.mp4']
        cond_videos = ['data/demo_video/hitting_bag.mp4']
        if args.multiple:
            # generate 100 for single bad example.
            split = args.split
            num_splits = args.total_split
            all_gen_dict = {}
            gen_cnt = 100
            dest = os.path.join(f'logs/{target_log_dir}', f'{gen_cnt}_times_split_{split}_wav_dict.pt')
            for i, v in enumerate(orig_videos):
                video_path = v
                start_sec = None
                if i % num_splits != split:
                    continue
                for j, ep in enumerate(cond_videos):
                    gen_audio_condImage_fast_multiple(video_path, ep, model, all_gen_dict,
                                                    spec_take_first=args.spec_take_first, gen_cnt=gen_cnt,
                                                    target_log_dir=target_log_dir, using_torch=True,
                                                    L=2.0, cond_cnt=j, style_transfer=style_transfer,
                                                    normalize=False, show_griffin_lim=False, vqgan_L=2.0)
            torch.save(all_gen_dict, dest)
        else:
            for i, v in enumerate(orig_videos.items()):
                    video_path = v
                    start_sec = None
                    for j, ep in enumerate(cond_videos):
                        gen_audio_condImage_fast(video_path, ep, model, spec_take_first=args.spec_take_first,
                                                target_log_dir=target_log_dir, using_torch=True,
                                                L=2.0, cond_cnt=j, style_transfer=style_transfer,
                                                normalize=False, show_griffin_lim=False, vqgan_L=2.0)
    elif args.countix_av_gen:
        # Update your own input and conditional videos here
        orig_videos = ['data/demo_video/tennis.mp4']
        cond_videos = ['data/demo_video/chopping.mp4']
        split = args.split
        all_gen_dict = {}
        gen_cnt = args.gen_cnt
        dest = os.path.join(f'logs/{target_log_dir}', f'{gen_cnt}_times_split_{split}_wav_dict.pt')
        if args.split == -1:
            W_scales = list(range(args.min_W_scale, args.max_W_scale+1))
        else:
            W_scales = [split + 1]
        for W_scale in W_scales:
            for orig_idx, tar_path in enumerate(orig_videos):
                start_sec = 1.0
                video_duration = get_duration(tar_path)
                if video_duration < 2 * W_scale:
                    continue
                ignore_input_spec = True
                for i, ep in enumerate(cond_videos):
                    ex_start_sec = 0.0
                    video_duration = get_duration(ep)
                    if video_duration < 2 * W_scale or tar_path == ep:
                        continue
                    if args.multiple:
                        gen_audio_condImage_fast_multiple(tar_path, ep, model, all_gen_dict,
                                            target_log_dir=target_log_dir, using_torch=True,
                                            L=2.0, cond_cnt=0, style_transfer=style_transfer,
                                            normalize=False, show_griffin_lim=False, vqgan_L=2.0,
                                            target_start_time=start_sec, cond_start_time=ex_start_sec, 
                                            remove_noise=True, W_scale=W_scale, gen_cnt=gen_cnt,
                                            slide_win_mode=slide_win_mode, temperature=args.temperature,
                                            ignore_input_spec=ignore_input_spec, tmp_path=tmp_path)
                    else:
                        gen_audio_condImage_fast(tar_path, ep, model,
                                        target_log_dir=target_log_dir, using_torch=True,
                                        L=2.0, cond_cnt=0, style_transfer=style_transfer,
                                        normalize=False, show_griffin_lim=False, vqgan_L=2.0,
                                        target_start_time=start_sec, cond_start_time=ex_start_sec, 
                                        remove_noise=True, W_scale=W_scale,
                                        slide_win_mode=slide_win_mode, temperature=args.temperature,
                                        ignore_input_spec=ignore_input_spec, tmp_path=tmp_path)
        if args.multiple:
            torch.save(all_gen_dict, dest)
    else:
        raise NotImplementedError
