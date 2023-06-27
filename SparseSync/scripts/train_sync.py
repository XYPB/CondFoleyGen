import torch
import torch.distributed as dist
from tqdm import tqdm
from utils.logger import LoggerWithTBoard

from scripts.train_utils import (EarlyStopper, apply_batch_mixup,
                                 broadcast_obj, get_batch_sizes, get_datasets,
                                 get_device, get_loaders, get_lr_scheduler,
                                 get_model, get_optimizer, get_transforms,
                                 init_ddp, is_master, load_ckpt,
                                 make_backward_and_optim_step, prepare_inputs,
                                 set_seed, toggle_mode, verbose_epoch_progress,
                                 verbose_iter_progress, verbose_lr,
                                 verbose_test_progress)


def train(cfg):
    init_ddp(cfg)
    global_rank = dist.get_rank() if dist.is_initialized() else cfg.training.global_rank
    # LoggerWithTBoard inherits Tensorboard summary module and, therefore, can be treated as one on steroids
    logger = LoggerWithTBoard(global_rank, cfg)

    set_seed(cfg.training.seed + global_rank)

    # makes iterations faster if your inputs are of a fixed size
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    torch.backends.cudnn.benchmark = True

    device, num_gpus = get_device(cfg)

    # ckpt_path was created only for the master (to keep it the same), now we broadcast it to each worker
    cfg.ckpt_path = broadcast_obj(cfg.ckpt_path, global_rank, device)
    # making sure each worker has the same ckpt path as the master
    assert hasattr(cfg, 'ckpt_path'), f'I AM AT RANK: {global_rank}'

    transforms = get_transforms(cfg)
    datasets = get_datasets(cfg, transforms)
    batch_sizes = get_batch_sizes(cfg, num_gpus)
    loaders = get_loaders(cfg, datasets, batch_sizes)
    model, model_without_ddp = get_model(cfg, device)
    optimizer = get_optimizer(cfg, model, num_gpus)
    lr_scheduler = get_lr_scheduler(cfg, optimizer)

    logger.log_param_num(global_rank, model)

    early_stopper = EarlyStopper(cfg.training.patience, cfg.training.to_max_metric, cfg.training.metric_name)

    # the scaller for the loss. Helps to avoid precision underflow during half prec training
    scaler = torch.cuda.amp.GradScaler()

    # this chunk has a complicate logic but it simply loads pre-trained ckpt during finetuning/resuming
    if cfg.training.run_test_only or cfg.training.resume or cfg.training.finetune:
        start_epoch, early_stopper.best_metric = load_ckpt(cfg, model_without_ddp, optimizer, scaler, lr_scheduler)
    else:
        start_epoch = 0

    # don't do training loops if a user wants to only probe the model on the test set
    num_epochs = 0 if cfg.training.run_test_only else cfg.training.num_epochs

    # loop over the train and validation multiple times (typical PT boilerplate)
    for epoch in range(start_epoch, num_epochs):

        phases_to_run_on = ['train', 'valid']
        if 'run_corrupted_val' not in cfg.training or cfg.training.run_corrupted_val:
            phases_to_run_on.extend(['valid_rand_aud', 'valid_rand_rgb', 'valid_perm_batch'])

        for phase in phases_to_run_on:
            # does model.eval() or .train() on appropriate submodules
            toggle_mode(cfg, model, phase)

            # init runnining results
            running_results = dict(logits=[], targets=[], loss_total=0)

            if dist.is_initialized():
                loaders[phase].sampler.set_epoch(epoch)

            # how many times to iterate through a evaluation se (makes estimates more robust for small dsets)
            if phase == 'valid' and 'VGGSoundSparsePicked' in loaders[phase].dataset.__class__.__name__:
                iter_times = cfg.data.dataset.params.get('iter_times', 1)
            else:
                iter_times = 1

            for it in range(iter_times):

                prog_bar = tqdm(loaders[phase], f'{phase} ({epoch})', ncols=0)
                for i, batch in enumerate(prog_bar):
                    # unfortunately, I had to use this to avoid GPU mem error on the second iteration
                    if i == 1:
                        torch.cuda.empty_cache()
                    iter_step = epoch * len(loaders[phase]) + i
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # sends targets and inputs to cuda
                    aud, vid, targets = prepare_inputs(batch, device, phase)

                    if phase == 'train':
                        aud = apply_batch_mixup(aud, cfg.training.mixup_alpha)

                    # gradient and half-precision toggles
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.autocast('cuda', enabled=cfg.training.use_half_precision):

                            # saves recontructed input to the model during the first iteration (detects bugs)
                            if is_master(global_rank) and iter_step == 0 and phase in ['train', 'valid']:
                                logger.vizualize_input(vid, aud, batch, iter_step)

                            loss, logits = model(vid, aud, targets)

                    if phase == 'train':
                        make_backward_and_optim_step(cfg, loss, model, optimizer, scaler, lr_scheduler)

                    # gathering results in one place to iterate on this later
                    iter_results = dict(
                        logits=[logits.detach().cpu()],
                        targets=[targets['offset_target'].cpu()],
                        loss_total=loss.item(),
                    )

                    if is_master(global_rank):
                        verbose_iter_progress(logger, prog_bar, iter_step, iter_results, phase)
                        if phase == 'train':
                            verbose_lr(logger, prog_bar, iter_step, lr_scheduler.get_last_lr()[0])

                    # doing it here instead of the dict() because we would like to verbose unscaled loss values
                    iter_results[f'loss_total'] /= len(loaders[phase])
                    iter_results[f'loss_total'] /= iter_times

                    # update running results
                    for k in running_results.keys():
                        running_results[k] += iter_results[k]

                if is_master(global_rank):
                    logger.print_logger.info(f'({phase}) Done {it} iterations out of {iter_times}')

            # logs epoch metrics to tensorboard/wandb
            epoch_loss, metrics = verbose_epoch_progress(global_rank, logger, running_results, phase, epoch)

            # Early stopping update
            if phase == cfg.training.early_stop_phase:
                has_model_improved = early_stopper.decide(global_rank, logger, metrics)
                if has_model_improved and is_master(global_rank):
                    # saves the best checkpoint. Replaces the previous one
                    logger.log_best_model(model_without_ddp, scaler, epoch_loss,
                                          epoch, optimizer, lr_scheduler, metrics, cfg)

            # wait for other workers to get here
            if dist.is_initialized():
                dist.barrier()

        if early_stopper.triggered:
            if is_master(global_rank):
                logger.print_logger.info(f'Training is early stopped @ {epoch}; RANK: {global_rank}')
            break

    if is_master(global_rank):
        logger.print_logger.info('Finished Training')

    # Testing the model
    phase = 'test'
    cfg.training.finetune = False
    # loading the best model
    ckpt_epoch, best_metric_val = load_ckpt(cfg, model_without_ddp, optimizer, scaler, lr_scheduler)
    if is_master(global_rank):
        logger.print_logger.info(f'Loading the best model from {cfg.ckpt_path}')
        logger.print_logger.info(f'Best metric: {best_metric_val}')
        logger.print_logger.info((f'The model was trained for {ckpt_epoch} epochs.'))
    model.eval()

    # init runnining results
    running_results = dict(logits=[], targets=[], loss_total=0)

    if dist.is_initialized():
        loaders[phase].sampler.set_epoch(ckpt_epoch)

    # how many times to iterate through a evaluation dataset (makes estimates more robust for small datasets)
    iter_times = cfg.data.dataset.params.get('iter_times', 1)
    for it in range(iter_times):
        prog_bar = tqdm(loaders[phase], f'{phase} ({ckpt_epoch})', ncols=0)
        for iter_step, batch in enumerate(prog_bar):
            # sends inputs and targets to cuda
            aud, vid, targets = prepare_inputs(batch, device, phase)
            # zero the parameter gradients
            optimizer.zero_grad()
            # gradient and half-precision toggles
            with torch.set_grad_enabled(False):
                with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
                    loss, logits = model(vid, aud, targets)

            # gathering results in one place to iterate on this later
            iter_results = dict(
                logits=[logits.detach().cpu()],
                targets=[targets['offset_target'].cpu()],
                loss_total=loss.item() / len(loaders[phase]) / iter_times,
            )
            for k in running_results.keys():
                running_results[k] += iter_results[k]

        if is_master(global_rank):
            logger.print_logger.info(f'Done {it} iterations out of {iter_times}')

    # logs test metrics to tensorboard/wandb
    verbose_test_progress(global_rank, logger, cfg, running_results, ckpt_epoch)
