import os
from omegaconf import OmegaConf
from scripts.train_utils import get_curr_time_w_random_shift
from utils.utils import cfg_sanity_check_and_patch
from scripts.train_feature_extractor import train as train_audio_feature_extractor
from scripts.train_sync import train as train_sync

def on_cluster():
    return 'SLURM_JOB_ID' in os.environ

def set_env_variables():
    # run sbatch with `--ntasks-per-node=GPUs`; MASTER_ADDR is expected to be `export`ed in sbatch file
    os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NPROCS']

def get_config():
    cfg_cli = OmegaConf.from_cli()
    cfg_yml = OmegaConf.load(cfg_cli.config)
    # the latter arguments are prioritized
    cfg = OmegaConf.merge(cfg_yml, cfg_cli)
    if 'start_time' not in cfg or cfg.start_time is None:
        cfg.start_time = get_curr_time_w_random_shift()
    # adds support for resolving `from_file:relative/path` config
    OmegaConf.register_new_resolver('from_file', lambda rel_path: OmegaConf.load(rel_path))
    OmegaConf.resolve(cfg)  # things like "${model.size}" in cfg will be resolved into values
    return cfg


def main(cfg):
    if cfg.action == 'train_audio_feature_extractor':
        train_audio_feature_extractor(cfg)
    elif cfg.action == 'train_avsync_model':
        cfg_sanity_check_and_patch(cfg)
        train_sync(cfg)
    # elif cfg.action == 'debug':
    #     cfg_sanity_check_and_patch(cfg)
    #     train_debug(cfg)


if __name__ == '__main__':
    cfg = get_config()

    if on_cluster():
        set_env_variables()

    main(cfg)
