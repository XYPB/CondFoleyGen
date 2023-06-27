import sys

import torch
import torch.nn as nn
from model.modules.feature_selector import CrossAttention

sys.path.insert(0, '.')  # nopep8
from model.modules.transformer import SelfAttention
from utils.utils import instantiate_from_config


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class Recorder(nn.Module):
    '''Adapted from lucidrains/vit-pytorch/'''
    def __init__(self, sync_model, device=None, watch_module=SelfAttention, submodule_name=None):
        super().__init__()
        self.transformer = sync_model
        if hasattr(self.transformer.transformer, 'config'):
            self.visual_block_shape = self.transformer.transformer.config.visual_block_shape
            self.audio_block_shape = self.transformer.transformer.config.audio_block_shape
        else:
            self.visual_block_shape = self.transformer.transformer.vis_pos_emb.block_shape
            self.audio_block_shape = self.transformer.transformer.aud_pos_emb.block_shape

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device
        self.watch_module = watch_module
        self.submodule_name = submodule_name

    def _hook(self, _, input, output):
        # self.recordings.append(output.clone().detach())
        self.recordings.append(input[0].clone().detach())

    def _register_hook(self):
        if self.submodule_name is None:
            source_module = self.transformer.transformer
        else:
            source_module = getattr(self.transformer.transformer, self.submodule_name)
        modules = find_modules(source_module, self.watch_module)
        for module in modules:
            handle = module.attn_drop.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.transformer

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def reorganize_self_att(self, att, vis_shape, aud_shape):
        att_dict = {}
        # (batch, layer, head, sequence_len, sequence_len)
        B, L, heads, S, S = att.shape
        tv, h, w = vis_shape
        f, ta = aud_shape
        # attention between all these keys with one another
        keys2shape = dict(OFF=[1], vis=vis_shape, MOD=[1], aud=aud_shape)

        # slices borders
        split_idx = [0, 1, 1+tv*h*w, 1+tv*h*w+1, 1+tv*h*w+1+f*ta]
        # pairwise grouping of attention
        att_dict = {}
        for k1, l1, r1 in zip(keys2shape.keys(), split_idx[:-1], split_idx[1:]):
            att_dict_k2s = {}
            for k2, l2, r2 in zip(keys2shape.keys(), split_idx[:-1], split_idx[1:]):
                a_temp = att.clone()[:, :, :, l1:r1, l2:r2]
                att_dict_k2s[k2] = a_temp.reshape(B, L, heads, *keys2shape[k1], *keys2shape[k2])
            att_dict[k1] = att_dict_k2s

        return att_dict

    def reorganize_cross_att(self, att, context_shape):
        # (batch, layer, head, selector_sequence, context_sequence)
        B, L, heads, Ss, Sc = att.shape
        att_dict = {self.submodule_name: {'context': att.reshape(B, L, heads, Ss, *context_shape)}}
        return att_dict

    def forward(self, *args):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        off_loss, off_logits = self.transformer(*args)

        # move all recordings to one device before stacking
        recordings = tuple(map(lambda t: t.to(torch.device('cpu')), self.recordings))
        if len(recordings) > 0:
            # (B, L, H, S, S) where S = 2 + tv*h*w + 1 + f*ta
            att = torch.stack(recordings, dim=1)
            if self.watch_module.__name__ == 'SelfAttention':
                att = self.reorganize_self_att(att, self.visual_block_shape, self.audio_block_shape)
            elif self.watch_module.__name__ == 'CrossAttention' and self.submodule_name == 'v_selector':
                att = self.reorganize_cross_att(att, self.visual_block_shape)
            elif self.watch_module.__name__ == 'CrossAttention' and self.submodule_name == 'a_selector':
                att = self.reorganize_cross_att(att, self.audio_block_shape)
            else:
                raise NotImplementedError(f'{self.submodule_name} doesnt have {self.watch_module.__name__}')
        else:
            att = None

        # return (cls_loss, off_loss), (cls_logits, off_logits), att
        return off_loss, off_logits, att


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from time import time

    cfg = OmegaConf.load('./configs/av_sync.yaml')
    cfg.training.use_half_precision = False

    device = torch.device('cuda:2')
    torch.cuda.set_device(device)

    # model = instantiate_from_config(cfg.model.params.transformer)
    model = instantiate_from_config(cfg.model)
    model = Recorder(model)
    model = model.to(device)

    start_time = time()
    for i in range(5):
        # vis = torch.rand(1, 50, 512, 7, 7, device=device)
        # aud = torch.rand(1, 512, 9, 27, device=device)
        vis = torch.rand(1, 50, 3, 224, 224, device=device)
        aud = torch.rand(1, 1, 257, 862, device=device)
        # cls_logits, off_logits, sync_logits, att = model(vis, aud)
        # inference in half precision
        with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
            cls_logits, off_logits, att = model(vis, aud)
    print('Time:', round(time() - start_time, 3))
