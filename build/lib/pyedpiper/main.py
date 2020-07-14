import copy
import _logging

from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from .modules.model_configurator import (
    DataloadersConfigurator,
    OptimizersConfigurator,
    EvaluationsConfigurator,
)

log = _logging.getLogger(__name__)


def configure_model(
        pl_model: LightningModule,
        config: DictConfig,
        dataloaders=True,
        epoch_evaluations=True,
        optimizers=True
):
    """
    Configure model with handy features to minimize boilerplate even more.
    todo: add examples for each shorthand

    :param pl_model:
    :param config:
    :param dataloaders:
    :param epoch_evaluations:
    :param optimizers:
    :return:
    """
    user_config = copy.deepcopy(config)
    pp_model = copy.deepcopy(pl_model)
    features = list()

    if dataloaders:
        features.append(DataloadersConfigurator(user_config))

    if optimizers:
        features.append(OptimizersConfigurator(user_config))

    if epoch_evaluations:
        features.append(EvaluationsConfigurator(user_config))

    # ...
    for feature in features:
        pp_model = feature.configure(pp_model)

    log.info('Model configured successfully.')

    return pp_model
