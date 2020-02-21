from core.common.config import Config
from core.common.modules.chain_function_applier import ChainFunctionApplier


class ConfigValidator:
    validations = []

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

    @classmethod
    def validate(cls, config: Config):
        validator = ChainFunctionApplier(cls.validations, Config)
        results = validator.apply(config)
        return results

    @classmethod
    def summary(cls, config: Config):
        results = cls.validate(config)
