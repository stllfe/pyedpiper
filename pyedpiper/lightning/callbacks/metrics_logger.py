from typing import List, Optional

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.metrics import Metric
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class MetricsLogger(Callback):

    def __init__(self,
                 metrics: List[Metric],
                 input_key: str,
                 target_key: str,
                 prefix: Optional[str] = None,
                 per_batch: bool = False):

        super(MetricsLogger, self).__init__()
        self.metrics = metrics

        self.input_key = input_key
        self.target_key = target_key

        self.per_batch = per_batch
        self.epoch_last_check = None

        # Initialize for storing values
        self.names = list(f"{str(prefix) + '/' if prefix else ''}{metric.name}" for metric in metrics)
        self.values = dict.fromkeys(self.names, 0.)

    def on_train_start(self, trainer, pl_module):

        if not trainer.logger:
            raise MisconfigurationException(
                "Cannot use MetricsLogger callback with Trainer that has no logger."
            )

    def on_train_end(self, trainer, pl_module):
        pass

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        epoch = trainer.current_epoch

        input = trainer.callback_metrics[self.input_key]
        target = trainer.callback_metrics[self.target_key]

        for name, metric in zip(self.names, self.metrics):
            self.values[name] = float(metric(input, target))

        if self._is_time_to_log(epoch):
            trainer.logger.log_metrics(self.values, step=epoch)

    def _is_time_to_log(self, epoch):
        # TODO: add `per_batch` or `period` evaluation
        if self.values:
            return True


