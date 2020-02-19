from core.common.config import Config
from core.common.modules.chain_function_applier import ChainFunctionApplier


class ConfigValidator(ChainFunctionApplier):
    custom_validations = []

    def __init__(self):
        super(ConfigValidator).__init__(self.custom_validations, Config)
