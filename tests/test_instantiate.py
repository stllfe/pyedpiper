from pyedpiper import instantiate
from pyedpiper.core.common import (
    _MODULE_KEY as MODULE_KEY,
    _PARAMS_KEY as PARAMS_KEY,
    _TARGET_KEY as TARGET_KEY,
)

THIS_MODULE = str(__name__)


class Simple:

    def __init__(self, a: int, b: str):
        self.a = a
        self.b = b


class SimpleWithDefaults:

    def __init__(self, a: int = 4, b: str = 'test'):
        self.a = a
        self.b = b


def build_config(target: type, params: dict):
    return {TARGET_KEY: target.__name__,
            MODULE_KEY: THIS_MODULE,
            PARAMS_KEY: params}


def test_simple_object():
    cfg = build_config(Simple, {'a': 10, 'b': 'test'})
    obj = instantiate(cfg)
    assert isinstance(obj, Simple)
    assert obj.a == 10
    assert obj.b == 'test'


def test_simple_object_with_defaults():
    # Some overrides
    cfg_1 = build_config(SimpleWithDefaults, {'a': 10})
    obj_1 = instantiate(cfg_1)

    assert isinstance(obj_1, SimpleWithDefaults)
    assert obj_1.a == 10
    assert obj_1.b == 'test'

    # No overrides
    cfg_2 = build_config(SimpleWithDefaults, params={})
    obj_2 = instantiate(cfg_2)

    assert isinstance(obj_2, SimpleWithDefaults)
    assert obj_2.a == 4
    assert obj_2.b == 'test'

    # All overrides
    cfg_3 = build_config(SimpleWithDefaults, {'a': 20, 'b': 'another'})
    obj_3 = instantiate(cfg_3)

    assert isinstance(obj_3, SimpleWithDefaults)
    assert obj_3.a == 20
    assert obj_3.b == 'another'
