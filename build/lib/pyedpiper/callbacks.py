import sys

from optuna import Trial
from optuna.exceptions import TrialPruned
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from _torch import Tensor
from tqdm import tqdm


class ProgressBar(Callback):
    """
    Global progress bar to replace the PyTorch Lightning's one.
    Just to have a bit more control over it.

    TODO: add progress bar for training, validation and testing loop.
    """

    def __init__(self, show: bool = True, leave: bool = True):
        super(ProgressBar, self).__init__()

        self.show = show
        self.desc = "Epoch {epoch}/{max_epoch} "
        self.leave = leave
        self.bar = None
        self.postfix = None

    def format_desc(self, trainer):
        return self.desc.format(epoch=trainer.current_epoch + 1,
                                max_epoch=trainer.max_epochs)

    def on_train_start(self, trainer, pl_module):
        self.bar = tqdm(
            desc=self.format_desc(trainer),
            total=trainer.max_epochs,
            leave=self.leave,
            disable=not self.show,
            file=sys.stdout,
        )

    def on_train_end(self, trainer: Trainer, pl_module):
        self.bar.close()
        self.bar = None

    def on_epoch_end(self, trainer: Trainer, pl_module):
        self.bar.set_description(self.format_desc(trainer))

        # Set metrics
        metrics = trainer.tqdm_metrics
        for k, v in metrics.items():
            if isinstance(v, Tensor):
                metrics[k] = v.squeeze().item()
        self.postfix = metrics

        # Update progress
        self.bar.set_postfix(self.postfix)
        self.bar.update()


class OptunaPruningCallback(EarlyStopping):
    """
    Callback for Optuna hyperparameter optimization library.
    """

    def __init__(self, trial: Trial, monitor: str):
        super(OptunaPruningCallback, self).__init__(monitor=monitor)

        self._trial = trial
        self._monitor = monitor

    def on_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        epoch = trainer.current_epoch
        current_score = logs.get(self._monitor)
        if current_score is None:
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise TrialPruned(message)


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback"""

    def __init__(self):
        super(MetricsCallback, self).__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

    def on_test_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)
