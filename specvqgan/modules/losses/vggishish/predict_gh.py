import os
import sys
import json
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from dataset import GreatestHit, AMT_test
import torch
import torch.nn as nn
from metrics import metrics
from omegaconf import OmegaConf
from model import VGGishish
from transforms import Crop, StandardNormalizeAudio, ToTensor


if __name__ == '__main__':
    cfg_cli = sys.argv[1]
    target_path = sys.argv[2]
    model_path = sys.argv[3]
    cfg_yml = OmegaConf.load(cfg_cli)
    # the latter arguments are prioritized
    cfg = cfg_yml
    OmegaConf.set_readonly(cfg, True)
    # print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    transforms = [
        StandardNormalizeAudio(cfg.mels_path),
    ]
    if cfg.cropped_size not in [None, 'None', 'none']:
        transforms.append(Crop(cfg.cropped_size))
    transforms.append(ToTensor())
    transforms = torchvision.transforms.transforms.Compose(transforms)

    testset = AMT_test(target_path, transforms, action_only=cfg.action_only, material_only=cfg.material_only)
    loader = DataLoader(testset, batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers, pin_memory=True)

    model = VGGishish(cfg.conv_layers, cfg.use_bn, num_classes=len(testset.label2target))
    ckpt = torch.load(model_path)['model']
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)

    model.eval()

    if cfg.cls_weights_in_loss:
        weights = 1 / testset.class_counts
    else:
        weights = torch.ones(len(testset.label2target))

    preds_from_each_batch = []
    file_path_from_each_batch = []
    for batch in tqdm(loader):
        inputs = batch['input'].to(device)
        file_path = batch['file_path']
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
        # for metrics calculation later on
        preds_from_each_batch += [outputs.detach().cpu()]
        file_path_from_each_batch += file_path
    preds_from_each_batch = torch.cat(preds_from_each_batch)
    _, preds = torch.topk(preds_from_each_batch, k=1)
    pred_dict = {fp: int(p.item()) for fp, p in zip(file_path_from_each_batch, preds)}
    mel_parent_dir = os.path.dirname(list(pred_dict.keys())[0])
    pred_list = [pred_dict[os.path.join(mel_parent_dir, f'{i}.npy')] for i in range(len(pred_dict))]
    json.dump(pred_list, open(target_path + f'_{cfg.exp_name}_preds.json', 'w'))
