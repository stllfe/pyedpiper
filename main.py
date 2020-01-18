from src.utils.helpers import (
    load_configuration,
    get_engines
)


def main():
    config = load_configuration()
    #engines = get_engines(config)

    print(config.custom)


if __name__ == '__main__':
    main()
