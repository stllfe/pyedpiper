from core.common.config import (
    Config,
    ConfigValidator,
    ConfigConfigurator,
)
from core.common.modules.closest_string_finder import ClosestStringFinder
from core.common.modules.module_loader import ModuleLoader
from core.common.modules.object_builder import ObjectBuilder


class Piper:
    def __init__(self, config: Config):
        self.config = Config.default().update(config)
        self.configurator = ConfigConfigurator()
        self.validator = ConfigValidator()

        self.module_loader = ModuleLoader()
        self.string_finder = ClosestStringFinder()
        self.object_builder = ObjectBuilder()

    def validate_config(self):
        pass