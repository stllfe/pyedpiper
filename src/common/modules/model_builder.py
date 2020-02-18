from src.common.config import Config
from src.common.model import DefaultModel


class ModelBuilder:
    """
    This class builds the main model to be used elsewhere in pipeline with the given config.
    Basically this thing wraps any model with suitable API for pytorch-lightning framework
    as well as sets all the basic functions for you.
    """

    # todo: implement me
    def __init__(self, config: Config):
        pass

    def build(self) -> DefaultModel:
        pass
