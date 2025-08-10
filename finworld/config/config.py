import os
from mmengine import Config as MMConfig
from argparse import Namespace

from dotenv import load_dotenv
load_dotenv(verbose=True)

from finworld.utils import assemble_project_path, get_tag_name, Singleton, set_seed

def check_level(level: str) -> bool:
    """
    Check if the level is valid.
    """
    valid_levels = ['1day', '1min', '5min', '15min', '30min', '1hour', '4hour']
    if level not in valid_levels:
        return False
    return True

def process_general(config: MMConfig) -> MMConfig:

    config.exp_path = assemble_project_path(os.path.join(config.workdir, config.tag))
    os.makedirs(config.exp_path, exist_ok=True)

    config.log_path = os.path.join(config.exp_path, getattr(config, 'log_path', 'finworld.log'))

    if "checkpoint_path" in config:
        config.checkpoint_path = os.path.join(config.exp_path, getattr(config, 'checkpoint_path', 'checkpoint'))
        os.makedirs(config.checkpoint_path, exist_ok=True)

    if "plot_path" in config:
        config.plot_path = os.path.join(config.exp_path, getattr(config, 'plot_path', 'plot'))
        os.makedirs(config.plot_path, exist_ok=True)

    if "tracker" in config:
        for key, value in config.tracker.items():
            config.tracker[key]['logging_dir'] = os.path.join(config.exp_path, value['logging_dir'])

    if "seed" in config:
        set_seed(config.seed)

    return config


class Config(MMConfig, metaclass=Singleton):
    def __init__(self):
        super(Config, self).__init__()

    def init_config(self, config_path: str, args: Namespace) -> None:
        # Initialize the general configuration
        mmconfig = MMConfig.fromfile(filename=assemble_project_path(config_path))
        if 'cfg_options' not in args or args.cfg_options is None:
            cfg_options = dict()
        else:
            cfg_options = args.cfg_options
        for item in args.__dict__:
            if item not in ['config', 'cfg_options'] and args.__dict__[item] is not None:
                cfg_options[item] = args.__dict__[item]
        mmconfig.merge_from_dict(cfg_options)

        tag = get_tag_name(
            tag=getattr(mmconfig, 'tag', None),
            assets_name=getattr(mmconfig, 'assets_name', None),
            source=getattr(mmconfig, 'source', None),
            data_type= getattr(mmconfig, 'data_type', None),
            level= getattr(mmconfig, 'level', None),
        )
        mmconfig.tag = tag

        # Process general configuration
        mmconfig = process_general(mmconfig)

        # Initialize the price downloader configuration
        if 'downloader' in mmconfig:
            if "assets_path" in mmconfig.downloader:
                mmconfig.downloader.assets_path = assemble_project_path(mmconfig.downloader.assets_path)
                assert check_level(mmconfig.downloader.level), f"Invalid level: {mmconfig.downloader.level}. Valid levels are: ['1day', '1min', '5min', '15min', '30min', '1hour', '4hour']"

        if 'processor' in mmconfig:
            if "assets_path" in mmconfig.processor:
                mmconfig.processor.assets_path = assemble_project_path(mmconfig.processor.assets_path)
            mmconfig.processor.repo_id = f"{os.getenv('HF_REPO_NAME')}/{mmconfig.processor.repo_id}"
            mmconfig.processor.repo_type = mmconfig.processor.repo_type if 'repo_type' in mmconfig.processor else 'dataset'

        self.__dict__.update(mmconfig.__dict__)

config = Config()