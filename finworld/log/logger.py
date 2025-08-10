import logging
import json
from enum import IntEnum
from typing import Optional, Any, Dict

from rich import box
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from finworld.utils import is_main_process
from finworld.utils import escape_code_brackets
from finworld.utils import Singleton
from finworld.log.wandb import wandb_logger
from finworld.log.tensorboard import tensorboard_logger


YELLOW_HEX = "#d4b702"

class LogLevel(IntEnum):
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    WARN = logging.WARN
    INFO = logging.INFO
    DEBUG = logging.DEBUG

class Logger(logging.Logger, metaclass=Singleton):
    def __init__(self, name="logger",
                 level=LogLevel.INFO):
        # Initialize the parent class
        super().__init__(name, level)

        # Define a formatter for log messages
        self.formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s:%(levelname)s - %(filename)s:%(lineno)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self.is_main_process = True

    def init_logger(self,
                    config,
                    level: int = LogLevel.INFO
                    ):
        """
        Initialize the logger with a file path and optional main process check.

        Args:
            log_path (str): The log file path.
            level (int, optional): The logging level. Defaults to logging.INFO.
            accelerator (Accelerator, optional): Accelerator instance to determine the main process.
        """
        log_path = config.log_path

        if "tracker" in config:
            wandb_logger.init_logger(config.tracker.wandb)
            tensorboard_logger.init_logger(config.tracker.tensorboard)
            self.wandb_logger = wandb_logger
            self.tensorboard_logger = tensorboard_logger
        else:
            self.wandb_logger = None
            self.tensorboard_logger = None

        self.is_main_process = is_main_process()

        self.handlers.clear()

        self.console = Console(
            width=None,
            markup=True,
            color_system="truecolor",
            force_terminal=True
        )
        rich_handler = RichHandler(
            console=self.console,
            rich_tracebacks=True,
            show_time=False,
            show_level=False,
            show_path=False,
            markup=True,
            omit_repeated_times=False
        )
        rich_handler.setLevel(level)
        rich_handler.setFormatter(self.formatter)
        self.addHandler(rich_handler)

        self.file_console = Console(
            width=None,
            markup=True,
            color_system="truecolor",
            force_terminal=True,
            file=open(log_path, "a", encoding="utf-8")
        )
        rich_file_handler = RichHandler(
            console=self.file_console,
            rich_tracebacks=True,
            show_time=False,
            show_level=False,
            show_path=False,
            markup=True,
            omit_repeated_times=False,
        )
        rich_file_handler.setLevel(level)
        rich_file_handler.setFormatter(self.formatter)
        self.addHandler(rich_file_handler)

        self.propagate = False

    def info(self, msg, *args, **kwargs):
        """
        Only for string messages, not for rich objects.
        """
        if not self.is_main_process:
            return
        kwargs.setdefault("stacklevel", 2)

        if "style" in kwargs:
            kwargs.pop("style")
        if "level" in kwargs:
            kwargs.pop("level")
        super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Only for string messages, not for rich objects.
        """
        if not self.is_main_process:
            return
        kwargs.setdefault("stacklevel", 2)
        super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if not self.is_main_process:
            return
        kwargs.setdefault("stacklevel", 2)
        super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if not self.is_main_process:
            return
        kwargs.setdefault("stacklevel", 2)
        super().critical(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        if not self.is_main_process:
            return
        kwargs.setdefault("stacklevel", 2)
        super().debug(msg, *args, **kwargs)

    def log(self,
            msg: Optional[Any] = None,
            level: LogLevel = LogLevel.INFO,
            **kwargs):
        """
        Log a rich object or a string message to both console and file.
        """
        if not self.is_main_process:
            return
        if isinstance(msg, str):
            self.info(msg, **kwargs)
        elif isinstance(msg, (Group, Panel, Rule, Syntax, Table, Tree)):
            self.console.print(msg, **kwargs)
            self.file_console.print(msg, **kwargs)

    def log_metric(self,
                   metric: Dict[str, Any],
                   step: Optional[int] = None,
                   level: LogLevel = LogLevel.INFO,
                   **kwargs
                   ):
        """
        Log a metric dictionary to the accelerator tracker.
        """
        if self.wandb_logger:
            self.wandb_logger.log(metric)
        if self.tensorboard_logger:
            for key, value in metric.items():
                self.tensorboard_logger.add_scalar(
                    tag=key,
                    scalar_value=value,
                    **kwargs
                )

    def log_error(self, error_message: str) -> None:
        self.log(escape_code_brackets(error_message), style="bold red", level=LogLevel.ERROR)

    def log_markdown(self, content: str, title: str | None = None, level=LogLevel.INFO, style=YELLOW_HEX) -> None:
        markdown_content = Syntax(
            content,
            lexer="markdown",
            theme="github-dark",
            word_wrap=True,
        )
        if title:
            self.info(
                Group(
                    Rule(
                        "[bold italic]" + title,
                        align="left",
                        style=style,
                    ),
                    markdown_content,
                ),
                level=level,
            )
        else:
            self.log(markdown_content, level=level)

    def log_code(self, title: str, content: str, level: int = LogLevel.INFO) -> None:
        self.info(
            Panel(
                Syntax(
                    content,
                    lexer="python",
                    theme="monokai",
                    word_wrap=True,
                ),
                title="[bold]" + title,
                title_align="left",
                box=box.HORIZONTALS,
            ),
            level=level,
        )

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        self.log(
            Rule(
                "[bold]" + title,
                characters="━",
                style=YELLOW_HEX,
            ),
            level=LogLevel.INFO,
        )

    def log_task(self, content: str, subtitle: str, title: str | None = None, level: LogLevel = LogLevel.INFO) -> None:
        self.log(
            Panel(
                f"\n[bold]{escape_code_brackets(content)}\n",
                title="[bold]New run" + (f" - {title}" if title else ""),
                subtitle=subtitle,
                border_style=YELLOW_HEX,
                subtitle_align="left",
            ),
            level=level,
        )

    def log_messages(self, messages: list[dict], level: LogLevel = LogLevel.DEBUG) -> None:
        messages_as_string = "\n".join([json.dumps(dict(message), indent=4, ensure_ascii=False) for message in messages])
        self.log(
            Syntax(
                messages_as_string,
                lexer="markdown",
                theme="github-dark",
                word_wrap=True,
            ),
            level=level,
        )

    def visualize_agent_tree(self, agent):
        def create_tools_section(tools_dict):
            table = Table(show_header=True, header_style="bold")
            table.add_column("Name", style="#1E90FF")
            table.add_column("Description")
            table.add_column("Arguments")

            for name, tool in tools_dict.items():
                args = [
                    f"{arg_name} (`{info.get('type', 'Any')}`{', optional' if info.get('optional') else ''}): {info.get('description', '')}"
                    for arg_name, info in getattr(tool, "inputs", {}).items()
                ]
                table.add_row(name, getattr(tool, "description", str(tool)), "\n".join(args))

            return Group("🛠️ [italic #1E90FF]Tools:[/italic #1E90FF]", table)

        def get_agent_headline(agent, name: str | None = None):
            name_headline = f"{name} | " if name else ""
            return f"[bold {YELLOW_HEX}]{name_headline}{agent.__class__.__name__} | {agent.model.model_id}"

        def build_agent_tree(parent_tree, agent_obj):
            """Recursively builds the agent tree."""
            parent_tree.add(create_tools_section(agent_obj.tools))

            if agent_obj.managed_agents:
                agents_branch = parent_tree.add("🤖 [italic #1E90FF]Managed agents:")
                for name, managed_agent in agent_obj.managed_agents.items():
                    agent_tree = agents_branch.add(get_agent_headline(managed_agent, name))
                    if managed_agent.__class__.__name__ == "CodeAgent":
                        agent_tree.add(
                            f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {managed_agent.additional_authorized_imports}"
                        )
                    agent_tree.add(f"📝 [italic #1E90FF]Description:[/italic #1E90FF] {managed_agent.description}")
                    build_agent_tree(agent_tree, managed_agent)

        main_tree = Tree(get_agent_headline(agent))
        if agent.__class__.__name__ == "CodeAgent":
            main_tree.add(
                f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {agent.additional_authorized_imports}"
            )
        build_agent_tree(main_tree, agent)
        self.log(main_tree)

logger = Logger()