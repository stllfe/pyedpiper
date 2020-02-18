from abc import abstractmethod, ABC
from copy import deepcopy

from src.common.config import Config
from src.common.modules.chain_function_applier import ChainFunctionApplier
from src.common.modules.config_validator import ConfigValidator


class BaseConfigConfigurator(ABC):
    def __init__(self, config: Config, validator: ChainFunctionApplier):
        self._base_config = deepcopy(config)
        self._validator = validator
        self._configured_config = None

    @property
    def config(self):
        return self._configured_config if self.is_configured else self._base_config

    @property
    def is_configured(self):
        return bool(self._configured_config)

    @abstractmethod
    def _configure_general(self):
        """
        Set devices, random seeds and other general parameters
        according to current `config.run` setting
        """
        pass

    @abstractmethod
    def _configure_train(self):
        """
        Necessary if `config.run` == 'train', optional otherwise
        """
        pass

    @abstractmethod
    def _configure_test(self):
        """
        Necessary if `config.run` == 'test', optional otherwise
        """
        pass

    @abstractmethod
    def _configure_data(self):
        """
        This is crucial since data configuration is needed anyways.
        If `config.run` == `train` and `validation` == True,
        then `data_val` (dataset folder) should be provided.
        """
        pass

    @abstractmethod
    def _configure_custom(self):
        """
        This can be set manually.
        """
        # todo: think of a good way to make it easily extendable in the future
        pass

    def configure(self) -> Config:
        self._configured_config = self._configure_general()
        self._configured_config = self._configure_data()
        self._configured_config = self._configure_train()
        self._configured_config = self._configure_test()
        return self._configured_config


class ConfigConfigurator(BaseConfigConfigurator):
    def __init__(self, config: Config):
        super(ConfigConfigurator).__init__(config, ConfigValidator())

    def _configure_general(self):
        # todo: start with implementing me!
        pass

    def _configure_train(self):
        pass

    def _configure_test(self):
        pass

    def _configure_data(self):
        pass

    def _configure_custom(self):
        pass
