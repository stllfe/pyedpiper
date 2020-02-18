import inspect
import logging
from typing import List


class ObjectBuilder:
    def __init__(self, config):
        self._config = config

    @staticmethod
    def _is_required(name, value, ignored_parameters=('self',)):
        return name not in ignored_parameters and value.default is inspect.Parameter.empty

    def _get_required_parameters(self, obj) -> List:
        required_parameters = list()
        signature = inspect.signature(obj.__init__)

        for name, value in signature.parameters.items():
            if self._is_required(name, value):
                required_parameters.append(name)

        return required_parameters

    def _build_obj_from_config(self, name, obj, **kwargs):
        required_parameters = self._get_required_parameters(obj)
        provided_parameters = self._config.keys()
        dependencies = dict(**kwargs)

        required_set = set(required_parameters)
        provided_set = set(provided_parameters)

        intersection = required_set.intersection(provided_set)
        if len(intersection) != len(required_set):
            error = "Not enough parameters for {} provided in config! ".format(name)
            message = "Required parameters are: {}".format(required_parameters)
            logging.error(error + message)
            raise Exception(error + message)
        try:
            parameters = {parameter: self._config[parameter] for parameter in required_parameters}
            parameters.update(dependencies)
            obj(**parameters)
        except ValueError as e:
            error = "Not enough parameters provided in kwargs! "
            message = "Original exception was: {}".format(e)
            logging.error(error + message)
            raise Exception(error + message)

    def build(self, obj_name, of_class, objects_map, **kwargs):
        if not objects_map.get(of_class):
            error = "{} of class `{}` is not implemented".format(obj_name, of_class)
            logging.error(error)
            raise NotImplementedError(error)
        else:
            obj = objects_map[of_class]

        return self._build_obj_from_config(name=of_class, obj=obj, **kwargs)
