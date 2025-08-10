import argparse
import os
import sys
from pathlib import Path
from mmengine import DictAction
import asyncio

root = str(Path(__file__).resolve().parents[2])
sys.path.append(root)

from finworld.log import logger
from finworld.config import config
from finworld.models import model_manager
from finworld.registry import TOOL, AGENT

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "llm_agent", "general.py"), help="config file path")

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
    args = parser.parse_args()
    return args

async def main():
    # Parse command line arguments
    args = parse_args()

    # Initialize the configuration
    config.init_config(args.config, args)

    # Initialize the logger
    logger.init_logger(log_path=config.log_path, accelerator=None)
    logger.info(f"| Logger initialized at: {config.log_path}")
    logger.info(f"| Config:\n{config.pretty_text}")

    # Registed models
    model_manager.init_models(use_local_proxy=False)
    logger.info("Registed models: %s", ", ".join(model_manager.registed_models.keys()))

    # Create agent
    agent_config = config.agent
    tools = [TOOL.build(tool) for tool in agent_config.tools]
    agent_config.update(
        {
            "template_path": agent_config.template_path,
            "model": model_manager.registed_models[agent_config.model],
            "tools": tools,
            "max_steps": agent_config.max_steps,
            "name": agent_config.name,
            "description": agent_config.description,
            "provide_run_summary": agent_config.provide_run_summary,
        }
    )
    agent = AGENT.build(agent_config)

    res = await agent.run("Use the python interpreter tool to calculate 2 + 3 and return the result.")
    logger.info(f"Result: {res}")

if __name__ == '__main__':
    asyncio.run(main())