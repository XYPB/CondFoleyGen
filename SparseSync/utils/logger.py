from matplotlib import pyplot as plt
import logging
import os
from pathlib import Path
from shutil import copytree, ignore_patterns, copy

import torch
from torchaudio.transforms import Spectrogram, GriffinLim
import torchvision
import wandb
from omegaconf import OmegaConf
from scripts.train_utils import get_curr_time_w_random_shift, is_master
from torch.utils.tensorboard import SummaryWriter, summary

from utils.utils import fix_prefix


class LoggerWithTBoard(SummaryWriter):

    def __init__(self, global_rank, cfg):
        # if not the master process, be silent and fail if mistakingly called
        if not is_master(global_rank):
            # just making a placeholder to broadcast to
            cfg.ckpt_path = None
            return None

        self.start_time = cfg.start_time
        self.logdir = os.path.join(cfg.logging.logdir, self.start_time)

        if not any([cfg.training.run_test_only, cfg.training.resume, cfg.training.finetune]):
            # self.logdir!
            cfg.ckpt_path = os.path.join(self.logdir, f'{self.start_time}.pt')

        self.ckpt_path = cfg.ckpt_path

        # weights and biases
        if cfg.logging.use_wandb:
            wandb.init(
                dir=cfg.logging.logdir,
                name=cfg.start_time,
                project=f'avsync-{cfg.action}',
                config=OmegaConf.to_container(cfg, resolve=[True | False]),
                sync_tensorboard=True
            )
            wandb_patterns_to_ignore = tuple(x.replace('*', '') for x in cfg.logging.patterns_to_ignore)
            wandb.run.log_code('.', exclude_fn=lambda x: x.endswith(wandb_patterns_to_ignore))
        # init tboard here because it also makes a directory
        super().__init__(self.logdir)

        if cfg.training.finetune:
            cfg.ckpt_path = copy(self.ckpt_path, os.path.join(self.log_dir, f'{self.start_time}.pt'))
            self.ckpt_path = cfg.ckpt_path
            print(f'Finetuning. The ckpt is copied to {self.ckpt_path}')

        now = get_curr_time_w_random_shift()

        # backup the cfg
        cfg_path = Path(self.log_dir) / f'cfg-{self.start_time}.yaml'
        # if exists, the fname will have the current time stamp
        if cfg_path.exists():
            cfg_path = cfg_path.parent / cfg_path.name.replace(self.start_time, now)
        OmegaConf.save(cfg, cfg_path)
        # backup the code state
        if cfg.logging.log_code_state:
            dest_dir = os.path.join(self.logdir, f'code-{self.start_time}')
            if not os.path.exists(dest_dir):
                copytree(os.getcwd(), dest_dir, ignore=ignore_patterns(*cfg.logging.patterns_to_ignore))

        # init logger which handles printing and logging mostly same things to the log file
        self.print_logger = logging.getLogger('main')
        self.print_logger.setLevel(logging.INFO)
        msgfmt = '[%(levelname)s] %(asctime)s - %(name)s \n    %(message)s'
        datefmt = '%d %b %Y %H:%M:%S'
        formatter = logging.Formatter(msgfmt, datefmt)
        # stdout
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)
        self.print_logger.addHandler(sh)
        # log file, if run for the second time, should append new logs
        fh = logging.FileHandler(os.path.join(self.log_dir, f'log-{self.start_time}.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.print_logger.addHandler(fh)

        self.print_logger.info(f'Saving logs and checkpoints @ {self.logdir}')

    def log_param_num(self, global_rank, model):
        if global_rank == 0:
            # for name, param in model.named_parameters():
            # if param.requires_grad:
            #     print(name, param.data.numel())
            param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_logger.info(f'The number of parameters: {param_num/1e+6:.3f} mil')
            self.add_scalar('num_params', param_num, 0)
            return param_num

    def log_iter_loss(self, loss, iter, phase, prefix: str = ''):
        self.add_scalar(f'{phase}/{fix_prefix(prefix)}loss_iter', loss, iter)

    def log_epoch_loss(self, loss, epoch, phase, prefix: str = ''):
        self.add_scalar(f'{phase}/{fix_prefix(prefix)}loss', loss, epoch)
        self.print_logger.info(f'{phase} ({epoch}): {fix_prefix(prefix)}loss {loss:.3f};')

    def log_iter_metrics(self, metrics_dict, epoch, phase, prefix: str = ''):
        for metric, val in metrics_dict.items():
            self.add_scalar(f'{phase}/{fix_prefix(prefix)}{metric}_iter', val, epoch)

    def log_epoch_metrics(self, metrics_dict, epoch, phase, prefix: str = ''):
        for metric, val in metrics_dict.items():
            self.add_scalar(f'{phase}/{fix_prefix(prefix)}{metric}', val, epoch)
        metrics_dict = {k: round(v, 4) for k, v in metrics_dict.items()}
        self.print_logger.info(f'{phase} ({epoch}) {fix_prefix(prefix)}metrics: {metrics_dict};')

    def log_test_metrics(self, metrics_dict, hparams_dict, best_epoch, prefix: str = ''):
        allowed_types = (int, float, str, bool, torch.Tensor)
        hparams_dict = {k: v for k, v in hparams_dict.items() if isinstance(v, allowed_types)}
        metrics_dict = {f'test/{fix_prefix(prefix)}{k}': round(v, 4) for k, v in metrics_dict.items()}
        exp, ssi, sei = summary.hparams(hparams_dict, metrics_dict)
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metrics_dict.items():
            self.add_scalar(k, v, best_epoch)
        self.print_logger.info(f'test ({best_epoch}) {fix_prefix(prefix)}metrics: {metrics_dict};')

    def log_best_model(self, model, scaler, loss, epoch, optimizer, lr_scheduler, metrics_dict, cfg):
        checkpoint = {
            'args': cfg,
            'loss': loss,
            'metrics': metrics_dict,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'scaler': scaler.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'model_type': model.__class__.__name__,
        }
        torch.save(checkpoint, self.ckpt_path)
        self.print_logger.info(f'Saved model in {self.ckpt_path}')

    def vizualize_input(self, vid: torch.Tensor, aud: torch.Tensor, batch, global_iter: int):
        ''' [B, Tv, C, H, W] [1, 1, F, Ta] '''
        B = vid.shape[0]
        aud_rec = aud.cpu()
        vid_rec = vid.cpu()
        # AudioStandardNormalize
        means = batch['meta']['spec_means'].view(B, 1, -1, 1)
        stds = batch['meta']['spec_stds'].view(B, 1, -1, 1)
        aud_rec = aud_rec * stds + means
        # AudioLog
        aud_rec = torch.exp(aud_rec)
        # AudioSpectrogram
        if not hasattr(self, 'griffinlim'):
            self.wav2spec = Spectrogram(n_fft=512, hop_length=128)
            self.griffinlim = GriffinLim(n_fft=512, hop_length=128)
        aud_rec = self.griffinlim(aud_rec)

        # RGBNormalize
        means = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        stds = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        vid_rec = vid_rec * stds + means
        # RGBToFloatToZeroOne
        vid_rec = (vid_rec * 255).short()

        vid_rec = vid_rec.permute(0, 1, 3, 4, 2)

        save_dir = os.path.join(self.logdir, 'viz')
        os.makedirs(save_dir, exist_ok=True)
        for b in range(len(aud_rec)):
            aud_rec_b = aud_rec[b]
            vid_rec_b = vid_rec[b]
            split = batch['split'][b]
            offset_sec = batch['targets']['offset_sec'][b]
            vfps = batch['meta']['video']['fps'][0][b]
            audio_fps = batch['meta']['audio']['framerate'][0][b]
            vid_id = Path(batch['path'][b]).stem
            save_path = Path(save_dir) / f'{str(global_iter).zfill(6)}_{split}_{vid_id}_{offset_sec:.2f}.mp4'
            torchvision.io.write_video(str(save_path), vid_rec_b, vfps.item(),
                                       audio_array=aud_rec_b, audio_fps=audio_fps.item(), audio_codec='aac')
            fig = plt.Figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 1, 1)
            spec_rec_b = self.wav2spec(aud_rec_b).permute(1, 2, 0).log().numpy()
            ax.imshow(spec_rec_b, cmap='gist_gray')
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            fig.savefig(str(save_path.with_suffix('.jpg')))
            # from scipy.io import wavfile
            # import numpy as np
            # wavfile.write(save_path.replace('mp4', 'wav'), 44100, aud.numpy().astype(np.float32))
