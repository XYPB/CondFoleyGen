import csv
import os
import sys
from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError:
    pass

import torch
import torchvision
import yaml
from omegaconf import OmegaConf

from specvqgan.util import get_ckpt_path

sys.path.insert(0, '.')  # nopep8
import soundfile

from feature_extraction.extract_mel_spectrogram import inv_transforms
from train import instantiate_from_config
from vocoder.modules import Generator


def load_model_from_config(config, sd, gpu=True, eval_mode=True, load_new_first_stage=False):
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params and not load_new_first_stage:
            config.params.first_stage_config.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if load_new_first_stage:
        first_stage_model = model.first_stage_model
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        try:
            print(f"Missing Keys in State Dict: {missing}")
            print(f"Unexpected Keys in State Dict: {unexpected}")
        except NameError:
            pass
    if load_new_first_stage:
        print('replace with new codebook model')
        model.first_stage_model = first_stage_model
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}

def load_vocoder(ckpt_vocoder: str, eval_mode: bool):
    ckpt_vocoder = Path(ckpt_vocoder)
    vocoder_sd = torch.load(ckpt_vocoder / 'best_netG.pt', map_location='cpu')

    with open(ckpt_vocoder / 'args.yml', 'r') as f:
        args = yaml.load(f, Loader=yaml.UnsafeLoader)

    vocoder = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers)
    vocoder.load_state_dict(vocoder_sd)

    if eval_mode:
        vocoder.eval()

    return {'model': vocoder}

def load_feature_extractor(gpu, eval_mode=True):
    s = '''
    feature_extractor:
        target: evaluation.feature_extractors.melception.Melception
        params:
            num_classes: 309
            features_list: ['logits']
            feature_extractor_weights_path: ./evaluation/logs/21-05-10T09-28-40/melception-21-05-10T09-28-40.pt
    transform_dset_out_to_inception_in:
    - target: evaluation.datasets.transforms.FromMinusOneOneToZeroOne
    - target: specvqgan.modules.losses.vggishish.transforms.StandardNormalizeAudio
      params:
        specs_dir: ./data/vggsound/melspec_10s_22050hz
        cache_path: ./specvqgan/modules/losses/vggishish/data/
    - target: evaluation.datasets.transforms.GetInputFromBatchByKey
      params:
        input_key: image
    - target: evaluation.datasets.transforms.ToFloat32'''
    feat_extractor_cfg = OmegaConf.create(s)
    # downloading the checkpoint for melception
    get_ckpt_path('melception', 'evaluation/logs/21-05-10T09-28-40')
    pl_sd = torch.load(feat_extractor_cfg.feature_extractor.params.feature_extractor_weights_path,
                       map_location="cpu")

    # use gpu=False to compute it on CPU
    feat_extractor = load_model_from_config(
        feat_extractor_cfg.feature_extractor, pl_sd['model'], gpu=gpu, eval_mode=eval_mode)['model']

    if feat_extractor_cfg.transform_dset_out_to_inception_in is not None:
        transforms = [instantiate_from_config(c) for c in feat_extractor_cfg.transform_dset_out_to_inception_in]
    else:
        transforms = [lambda x: x]
    transforms = torchvision.transforms.Compose(transforms)

    vggsound_meta = list(csv.reader(open('./data/vggsound.csv'), quotechar='"'))
    unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
    label2target = {label: target for target, label in enumerate(unique_classes)}
    target2label = {target: label for label, target in label2target.items()}
    return {'model': feat_extractor, 'transforms': transforms, 'target2label': target2label}


def show_wave_in_streamlit(wave_npy, sample_rate, caption):
    # showing in streamlit. We cannot just show the npy wave and we need to save it first
    temp_wav_file_path = 'todel.wav'
    soundfile.write(temp_wav_file_path, wave_npy, sample_rate, 'PCM_24')
    st.text(caption)
    st.audio(temp_wav_file_path, format='audio/wav')
    os.remove(temp_wav_file_path)

def spec_to_audio_to_st(x, spec_dir_path, sample_rate, show_griffin_lim, vocoder=None, show_in_st=True):
    # audios are in [-1, 1], making them in [0, 1]
    spec = (x.data.squeeze(0) + 1) / 2

    out = {}
    if vocoder:
        # (L,) <- wave: (1, 1, L).squeeze() <- spec: (1, F, T)
        wave_from_vocoder = vocoder(spec).squeeze().cpu().numpy()
        out['vocoder'] = wave_from_vocoder
        if show_in_st:
            show_wave_in_streamlit(wave_from_vocoder, sample_rate, 'Reconstructed Wave via MelGAN')

    if show_griffin_lim:
        spec = spec.squeeze(0).cpu().numpy()
        wave_from_griffinlim = inv_transforms(spec, Path(spec_dir_path).stem)
        out['inv_transforms'] = wave_from_griffinlim
        if show_in_st:
            show_wave_in_streamlit(wave_from_griffinlim, sample_rate, 'Reconstructed Wave via Griffin Lim')

    return out
