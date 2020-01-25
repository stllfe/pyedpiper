from src.utils.helpers import (
    load_configuration,
    get_engines
)
from src.custom import XceptionModel


def main():
    config = load_configuration()
    print(config.custom, config.train)


if __name__ == '__main__':
    main()
