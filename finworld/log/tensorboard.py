from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Any

from finworld.utils import Singleton
from finworld.utils import is_main_process

__all__= [
    'TensorboardLogger',
    'tensorboard_logger',
]

class TensorboardLogger(SummaryWriter, metaclass=Singleton):
    def __init__(self):
        self.accelerator = None
        self.is_main_process = True

    def init_logger(self, config):
        """
        Initialize the logger with a file path and optional main process check.

        Args:
            log_path (str): The log file path.
            level (int, optional): The logging level. Defaults to logging.INFO.
            accelerator (Accelerator, optional): Accelerator instance to determine the main process.
        """

        self.is_main_process = is_main_process()

        logging_dir = config['logging_dir']

        if self.is_main_process:
            super().__init__(logging_dir)
        else:
            self.file_writer = None

    def add_scalar(
            self,
            tag,
            scalar_value,
            global_step=None,
            walltime=None,
            new_style=False,
            double_precision=False,
        ):
        """
        Add a scalar value to the logs.
        """
        if not self.is_main_process:
            return

        super().add_scalar(tag,
                           scalar_value,
                           global_step,
                           walltime,
                           new_style,
                           double_precision)

    def add_image(self,
                  tag,
                  img_tensor,
                  global_step=None,
                  walltime=None,
                  dataformats='CHW'):
        """
        Add an image to the logs.
        """
        if not self.is_main_process:
            return

        super().add_image(tag, img_tensor, global_step, walltime, dataformats)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """
        Add text to the logs.
        """
        if not self.is_main_process:
            return
        super().add_text(tag, text_string, global_step, walltime)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        """
        Add a histogram to the logs.
        """
        if not self.is_main_process:
            return
        super().add_histogram(tag, values, global_step, bins, walltime, max_bins)

    def add_graph(self, model,
                  input_to_model=None,
                  verbose=False,
                  use_strict_trace=True):
        """
        Add a graph to the logs.
        """
        if not self.is_main_process:
            return
        super().add_graph(model, input_to_model, verbose, use_strict_trace)

    def flush(self):
        """
        Flush the logs to disk.
        """
        if not self.is_main_process:
            return
        super().flush()

    def close(self):
        """
        Close the logger to release resources.
        """
        if not self.is_main_process:
            return
        super().close()

tensorboard_logger = TensorboardLogger()