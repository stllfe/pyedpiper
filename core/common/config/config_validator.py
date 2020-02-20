from core.common.config import Config
from core.common.modules.chain_function_applier import ChainFunctionApplier


class ConfigValidator:
    validations = []

    def __init__(self):
        self.validator = ChainFunctionApplier(self.validations, Config)

    @staticmethod
    def is_applicable_validation(function):
        if ChainFunctionApplier.is_applicable(function):
            try:
                empty_config = Config()
                result = function(empty_config)
                return isinstance(result, list)
            except Exception:
                pass
        return False

    def validate(self, config: Config):
        results = self.validator.apply(config)
        return results
