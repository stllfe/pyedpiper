from core.common.config.config import load_configuration
from core.common.config.config_configurator import ConfigConfigurator
from core.common.modules.model_builder import ModelBuilder


def main():
    config = ConfigConfigurator(load_configuration()).configure()
    model = ModelBuilder(config).build()
    model.run()
    print()


if __name__ == '__main__':
    main()
