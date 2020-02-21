from abc import abstractmethod, ABC
from copy import deepcopy

from core.common.config import Config


class BaseConfigConfigurator(ABC):
    @abstractmethod
    def _configure_general(self, config: Config):
        """
        Set devices, random seeds and other general parameters
        according to current `config.run` setting
        """
        pass

    @abstractmethod
    def _configure_train(self, config: Config):
        """
        Necessary if `config.run` == 'train', optional otherwise
        """
        pass

    @abstractmethod
    def _configure_test(self, config: Config):
        """
        Necessary if `config.run` == 'test', optional otherwise
        """
        pass

    @abstractmethod
    def _configure_data(self, config: Config):
        """
        This is crucial since data configuration is needed anyways.
        If `config.run` == `train` and `validation` == True,
        then `data_val` (dataset folder) should be provided.
        """
        pass

    @abstractmethod
    def _configure_custom(self, config: Config):
        """
        This can be set manually.
        """
        # todo: think of a good way to make it easily extendable in the future
        pass

    def configure(self, config: Config) -> Config:
        base_config = deepcopy(config)
        config = self._configure_general(base_config)
        config = self._configure_data(config)
        config = self._configure_train(config)
        config = self._configure_test(config)
        return config


class ConfigConfigurator(BaseConfigConfigurator):

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
