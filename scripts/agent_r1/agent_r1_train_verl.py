import os
import argparse
import sys
from pathlib import Path
import ray
from omegaconf import OmegaConf
from mmengine import DictAction

from dotenv import load_dotenv
load_dotenv(verbose=True)

ROOT = str(Path(__file__).resolve().parents[2])
CUR = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CUR)


from finworld.config import build_config
from finworld.utils import assemble_project_path
from finworld.multi_verl.agent_rl.trainer.main_agent import run_agent

def get_args_parser():
    parser = argparse.ArgumentParser(description="Train script for finworld")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "agent_r1", "verl_multigpu_grpo.py"), help="config file path")
    parser.add_argument("--if_remove", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    return parser

def main(args):
    # 1. build config
    config = build_config(assemble_project_path(args.config), args)
    cfg_dict = config.to_dict()
    config = OmegaConf.create(cfg_dict)

    # 2. run agent
    run_agent(config)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

