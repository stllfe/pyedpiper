import _common
import _logging
from abc import ABC, abstractmethod

import _torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from ..utils import instantiate, average_metrics, average_and_log_metrics

log = _logging.getLogger(__name__)


class ModelInspectorMixin(ABC):

    @staticmethod
    def is_function_implemented(f_name, model: LightningModule):
        """
        Convenience method to tell if wrapped model has such method.
        Original method is taken from 'pytorch_lightning.Trainer' class.
        """
        f_op = getattr(model, f_name, None)
        return callable(f_op)

    @staticmethod
    def is_overriden(f_name, model: LightningModule):
        """
        Convenience method to tell if wrapped model has its inherited method overriden.
        Original method is taken from 'pytorch_lightning.Trainer' class.
        """
        super_object = LightningModule

        # when code pointers are different, it was overriden
        is_overriden = getattr(model, f_name).__code__ is not getattr(super_object, f_name).__code__
        return is_overriden


class BaseConfigurator(ModelInspectorMixin):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def configure(self, model: LightningModule) -> LightningModule:
        pass


class DataloadersConfigurator(BaseConfigurator):
    """
    If you want to use this feature implement 'configure_datasets' method in your model.
    It should return 'dict' object with keys for every mode you wish to run.
    For example:
    ...
    def configure_datasets(self):
        from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor

        train_transforms = Compose([RandomHorizontalFlip(), ToTensor()])
        test_transforms = ToTensor()

        train_dataset = instantiate(self.config.dataset.train, transform=train_transforms)
        test_dataset = instantiate(self.config.dataset.test, transform=test_transforms)
        val_dataset = instantiate(self.config.dataset.val, transform=test_transforms)

        return {'train': train_dataset, 'test': test_dataset, 'val': val_dataset}
    """

    def is_available(self, model: LightningModule):
        return self.is_function_implemented('configure_datasets', model)

    @staticmethod
    def _validate_datasets(datasets):
        if not datasets:
            error = "Model's 'configure_datasets' method returned 'None'."
            log.error(error)
            raise AttributeError(error)

        if not isinstance(datasets, dict) or not issubclass(type(datasets), dict):
            error = ("Datasets returned by 'configure_datasets' method should contain dict or dict-like object."
                     "Got '{}' instead".format(type(datasets)))

            log.error(error)
            raise TypeError(error)

        if len(datasets) > 3:
            error = "Datasets dict returned by 'configure_datasets' has unexpected len of {}".format(len(datasets))
            log.error(error)
            raise ValueError(error)

        for mode in datasets.keys():
            possible_modes = ['train', 'val', 'test']
            if mode not in possible_modes:
                error = "Unexpected mode '{}' found in 'datasets'. Use one of {}".format(mode, possible_modes)
                log.error(error)
                raise ValueError(error)

    def _build_dataloader(self, dataset):
        return instantiate(self.config.dataloader, dataset=dataset)

    def configure(self, model: LightningModule):
        if not self.is_available(model):
            warning = "Piper can't configure dataloaders. \n{}".format(self.__doc__)
            log.warning(warning)
            raise RuntimeWarning

        datasets = model.configure_datasets()
        self._validate_datasets(datasets)
        for mode, dataset in datasets.items():
            method_name = f'{mode}_dataloader'
            if self.is_overriden(method_name, model):
                log.info("'{}' is overriden by model. Skipping automatic configuration ...".format(method_name))
                continue
            setattr(model, method_name, lambda: self._build_dataloader(dataset))
        return model


class OptimizersConfigurator(BaseConfigurator):
    def _configure_optimizers(self, model: LightningModule):
        optimizer = instantiate(self.config.optimizer, params=model.parameters())

        if self.config.get('scheduler'):
            scheduler = instantiate(self.config.scheduler, optimizer=optimizer)
            return [optimizer], [scheduler]

        return optimizer

    def configure(self, model: LightningModule) -> LightningModule:
        if self.is_overriden('configure_optimizers', model):
            log.info("'configure_optimizers' is implemented by model. Skipping configuration ...")
            return model

        if 'optimizer' not in self.config:
            error = ("Can't to automatically build optimizers and schedulers",
                     "No such config keys were provided.")

            log.error(error)
            raise ValueError(error)

        setattr(model, 'configure_optimizers', lambda: self._configure_optimizers(model))
        return model


class EvaluationsConfigurator(BaseConfigurator):

    def is_available(self, model: LightningModule):
        return self.is_overriden('validation_step', model)

    def configure(self, model: LightningModule) -> LightningModule:
        if not self.is_overriden('validation_step', model):
            log.info("Model's 'validation_step' is not implemented. automatic 'validation_epoch_end' configuration ...")
            return model

        for step in ['validation', 'test']:
            if not self.is_overriden('%s_step' % step, model):
                message = "Model's '{}_step' is not implemented. Skipping automatic '{}_epoch_end' configuration ..."
                log.info(message.format(step, step))
                continue

            if self.is_overriden('%s_epoch_end' % step, model):
                message = "Model's '{}_epoch_end' is already implemented. Skipping configuration ..."
                log.info(message.format(step))
                continue

            if self.config.piper.log.tensorboard or self.config.piper.log.csv:
                setattr(model, '%s_epoch_end' % step, lambda outputs: average_and_log_metrics(outputs))
            else:
                setattr(model, '%s_epoch_end' % step, lambda outputs: average_metrics(outputs))

        return model


if __name__ == "__main__":
    outputs = [
        {'test_loss': _common.as_tensor(1).float(), 'progress_bar': {'test_acc': _common.as_tensor(0.3).float()}},
        {'test_loss': _common.as_tensor(2).float(), 'progress_bar': {'test_acc': _common.as_tensor(0.6).float()}},
        {'test_loss': _common.as_tensor(0.4).float(), 'progress_bar': {'test_acc': _common.as_tensor(0.3).float()}},
        {'test_loss': _common.as_tensor(0.2).float(), 'progress_bar': {'test_acc': _common.as_tensor(0.1).float()}},
        {'test_loss': _common.as_tensor(0.1).float(), 'progress_bar': {'test_acc': _common.as_tensor(1).float()}}
    ]

    new_dict = average_and_log_metrics(outputs)
    print(new_dict)
    print(_common.as_tensor([1, 2, 0.4, 0.2, 0.1]).mean(), _common.as_tensor([0.3, 0.6, 0.3, 0.1, 1]).mean())
