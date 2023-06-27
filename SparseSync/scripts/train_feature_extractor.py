import sys

import torch
import torchvision

sys.path.append('.')  # nopep8
import torch.distributed as dist

from dataset.vggsound import VGGSoundAudioOnly
from model.modules.feat_extractors.loss import WeightedCrossEntropy
from model.modules.feature_extractors import ResNetAudio
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from utils.logger import LoggerWithTBoard
from utils.utils import instantiate_from_config

from scripts.train_utils import get_lr_scheduler, init_ddp, make_backward_and_optim_step, set_seed, calc_metrics


def train(cfg):
    init_ddp(cfg)
    if dist.get_world_size() > 1:
        raise NotImplementedError('ddp is not supported')
    global_rank = dist.get_rank() if dist.is_initialized() else cfg.training.global_rank
    logger = LoggerWithTBoard(global_rank, cfg)
    set_seed(cfg.training.seed + global_rank)

    meta_path = './data/vggsound.csv'
    splits_path = './data/'

    transform_sequence_train_cfg = cfg.get('transform_sequence_train', None)
    transform_sequence_test_cfg = cfg.get('transform_sequence_test', None)

    if transform_sequence_train_cfg is None:
        transforms_train = [lambda x: x]
    else:
        transforms_train = [instantiate_from_config(c) for c in transform_sequence_train_cfg]

    if transform_sequence_test_cfg is None:
        transforms_test = [lambda x: x]
    else:
        transforms_test = [instantiate_from_config(c) for c in transform_sequence_test_cfg]

    transforms_train = torchvision.transforms.Compose(transforms_train)
    transforms_test = torchvision.transforms.Compose(transforms_test)

    datasets = {
        'train': VGGSoundAudioOnly('train', cfg.data.vids_dir, transforms_train,
                                   cfg.data.to_filter_bad_examples, splits_path, meta_path),
        'valid': VGGSoundAudioOnly('valid', cfg.data.vids_dir, transforms_test,
                                   cfg.data.to_filter_bad_examples, splits_path, meta_path),
        'test': VGGSoundAudioOnly('test', cfg.data.vids_dir, transforms_test,
                                  cfg.data.to_filter_bad_examples, splits_path, meta_path),
    }

    loaders = {
        'train': DataLoader(datasets['train'], batch_size=cfg.training.batch_size, shuffle=True, drop_last=True,
                            num_workers=cfg.training.num_workers, pin_memory=True),
        'valid': DataLoader(datasets['valid'], batch_size=cfg.training.batch_size*2,
                            num_workers=cfg.training.num_workers, pin_memory=True),
        'test': DataLoader(datasets['test'], batch_size=cfg.training.batch_size*2,
                           num_workers=cfg.training.num_workers, pin_memory=True),
    }
    device = torch.device(cfg.training.local_rank)

    model = ResNetAudio(cfg.model, num_classes=cfg.training.num_classes, extract_features=False)
    model = model.to(device)

    param_num = logger.log_param_num(global_rank, model)

    eps = 1e-7 if cfg.training.use_half_precision else 1e-8
    if cfg.training.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), cfg.training.learning_rate,
                                     cfg.training.optimizer.betas, eps, cfg.training.optimizer.weight_decay)
    elif cfg.training.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), cfg.training.learning_rate,
                                      cfg.training.optimizer.betas, eps, cfg.training.optimizer.weight_decay)
    elif cfg.training.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), cfg.training.learning_rate,
                                    cfg.training.optimizer.momentum,
                                    weight_decay=cfg.training.optimizer.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer: "{cfg.training.optimizer.name}" is not implemented')

    if cfg.training.cls_weights_in_loss:
        weights = 1 / datasets['train'].class_counts
    else:
        weights = torch.ones(len(datasets['train'].target2label))
    criterion = WeightedCrossEntropy(weights.to(device))

    # the scaller for the loss. Helps to avoid precision underflow during half prec training
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = get_lr_scheduler(cfg, optimizer)

    # loop over the train and validation multiple times (typical PT boilerplate)
    start_epoch = 0
    no_change_epochs = 0
    best_valid_loss = float('inf')
    early_stop_triggered = False

    num_epochs = cfg.training.num_epochs
    if cfg.training.run_test_only or cfg.training.resume:
        ckpt = torch.load(cfg.ckpt_path, map_location=torch.device('cpu'))
        optimizer.load_state_dict(ckpt['optimizer'])
        model.load_state_dict(ckpt['model'])
        scaler.load_state_dict(ckpt['scaler'])
        if cfg.training.resume:
            start_epoch = ckpt['epoch'] + 1
        else:
            num_epochs = 0

    for epoch in range(start_epoch, num_epochs):

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0
            preds_from_each_batch = []
            targets_from_each_batch = []

            prog_bar = tqdm(loaders[phase], f'{phase} ({epoch})', ncols=0)
            for i, batch in enumerate(prog_bar):
                inputs = batch['audio'].to(device)
                targets = batch['target'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
                        # inception v3
                        if phase == 'train':
                            outputs = model(inputs)
                            # always weights even if they are equal
                            loss = criterion(outputs, targets, to_weight=True)
                        else:
                            outputs = model(inputs)
                            # during eval time the loss will not be weighted to depict the reality
                            loss = criterion(outputs, targets, to_weight=False)

                if phase == 'train':
                    make_backward_and_optim_step(cfg, loss, model, optimizer, scaler, lr_scheduler)

                # loss
                running_loss += loss.item()

                # for metrics calculation later on
                preds_from_each_batch += [outputs.detach().cpu()]
                targets_from_each_batch += [targets.cpu()]

                # iter logging
                if i % 50 == 0:
                    logger.log_iter_loss(loss.item(), epoch*len(loaders[phase])+i, phase)
                    # tracks loss in the tqdm progress bar
                    prog_bar.set_postfix(loss=loss.item())

            # logging loss
            epoch_loss = running_loss / len(loaders[phase])
            logger.log_epoch_loss(epoch_loss, epoch, phase)

            # logging metrics
            preds_from_each_batch = torch.cat(preds_from_each_batch).float()
            targets_from_each_batch = torch.cat(targets_from_each_batch).long()
            metrics_dict = calc_metrics(targets_from_each_batch, preds_from_each_batch)
            logger.log_epoch_metrics(metrics_dict, epoch, phase)

            # Early stopping
            if phase == 'valid':
                if epoch_loss < best_valid_loss:
                    no_change_epochs = 0
                    best_valid_loss = epoch_loss
                    logger.log_best_model(model, scaler, epoch_loss, epoch, optimizer, lr_scheduler,
                                          metrics_dict, cfg)
                else:
                    no_change_epochs += 1
                    logger.print_logger.info(
                        f'Valid loss hasnt changed for {no_change_epochs} patience: {cfg.training.patience}'
                    )
                    if no_change_epochs >= cfg.training.patience:
                        early_stop_triggered = True

        if early_stop_triggered:
            logger.print_logger.info(f'Training is early stopped @ {epoch}')
            break

    logger.print_logger.info('Finished Training')

    # loading the best model
    ckpt_path = cfg.ckpt_path
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    logger.print_logger.info(f'Loading the best model from {ckpt_path}')
    logger.print_logger.info((f'The model was trained for {ckpt["epoch"]} epochs. Loss: {ckpt["loss"]:.4f}'))

    # Testing the model
    model.eval()
    running_loss = 0
    preds_from_each_batch = []
    targets_from_each_batch = []

    prog_bar = tqdm(loaders['test'], f'test ({ckpt["epoch"]})', ncols=0)
    for i, batch in enumerate(prog_bar):
        inputs = batch['audio'].to(device)
        targets = batch['target'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        with torch.set_grad_enabled(False):
            with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
                outputs = model(inputs)
                loss = criterion(outputs, targets, to_weight=False)

        # loss
        running_loss += loss.item()

        # for metrics calculation later on
        preds_from_each_batch += [outputs.detach().cpu()]
        targets_from_each_batch += [targets.cpu()]

    # logging metrics
    preds_from_each_batch = torch.cat(preds_from_each_batch).float()
    targets_from_each_batch = torch.cat(targets_from_each_batch).long()
    test_metrics_dict = calc_metrics(targets_from_each_batch, preds_from_each_batch)
    test_metrics_dict['avg_loss'] = running_loss / len(loaders['test'])
    test_metrics_dict['param_num'] = param_num
    # TODO: I have no idea why tboard doesn't keep metrics (hparams) when
    # I run this experiment from cli: `python train_melception.py config=./configs/vggish.yaml`
    # while when I run it in vscode debugger the metrics are logger (wtf)
    logger.log_test_metrics(test_metrics_dict, dict(cfg), ckpt['epoch'])

    logger.print_logger.info('Finished the experiment')


# if __name__ == '__main__':
#     from omegaconf import OmegaConf
#     cfg_cli = OmegaConf.from_cli()
#     if len(cfg_cli) == 0:
#         cfg_cli = OmegaConf.create()
#         cfg_cli.config = './configs/audio_feature_extractor.yaml'
#     cfg_yml = OmegaConf.load(cfg_cli.config)
#     # the latter arguments are prioritized
#     cfg = OmegaConf.merge(cfg_yml, cfg_cli)
#     cfg.data.vids_dir = 'PLACEHOLDER'
#     OmegaConf.set_readonly(cfg, True)
#     print(OmegaConf.to_yaml(cfg))

#     train(cfg)
