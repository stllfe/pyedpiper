from core.common.config import Config
from core.common.decorators import ignore
from core.common.modules.chain_function_applier import ChainFunctionApplier


class ConfigValidator:
    validations = []

    @staticmethod
    @ignore(Exception, default=False)
    def is_applicable_validation(function):
        if ChainFunctionApplier.is_applicable(function):
            empty_config = Config()
            result = function(empty_config)
            return isinstance(result, list)
        return False

    @classmethod
    def validate(cls, config: Config):
        validator = ChainFunctionApplier(cls.validations, Config)
        results = validator.apply(config)
        return results

    @classmethod
    def summary(cls, config: Config):
        results = cls.validate(config)
        pass
