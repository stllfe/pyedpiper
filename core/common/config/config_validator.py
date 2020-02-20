from core.common.config import Config
from core.common.modules.chain_function_applier import ChainFunctionApplier


class ConfigValidator:
    validations = []

    def __init__(self):
        self.validator = ChainFunctionApplier(self.validations, Config)

    def validate(self, config: Config):
        results = self.validator.apply(config)
        pass
