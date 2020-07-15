from typing import List, Optional

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.metrics import Metric
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class MetricsLogger(Callback):

    def __init__(self, metrics: List[Metric],
                 outputs_key: str,
                 targets_key: str,
                 prefix: Optional[str] = None,
                 per_batch: bool = False,
                 allow_nokey=False,):

        super(MetricsLogger, self).__init__()
        self.metrics = metrics

        self.outputs_key = outputs_key
        self.targets_key = targets_key

        self.per_batch = per_batch
        self.epoch_last_check = None
        self.prefix = prefix or ""
        self.allow_nokey = allow_nokey

        # Initialize for storing values
        self.names = list(f"{prefix}/{metric.name}".strip("/") for metric in metrics)
        self.values = dict.fromkeys(self.names, 0.)

    def on_train_start(self, trainer, pl_module):

        if not trainer.logger:
            raise MisconfigurationException(
                "Cannot use MetricsLogger callback with Trainer that has no logger."
            )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        epoch = trainer.current_epoch
        outputs = trainer.callback_metrics[self.outputs_key]
        targets = trainer.callback_metrics[self.targets_key]

        for name, metric in zip(self.names, self.metrics):
            self.values[name] = float(metric(outputs, targets))

        if self._is_time_to_log(epoch):
            trainer.logger.log_metrics(self.values, step=epoch)

    def _is_time_to_log(self, epoch):
        # TODO: add `per_batch` or `period` evaluation
        if self.values:
            return True


