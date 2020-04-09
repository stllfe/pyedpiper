import inspect
import logging
from typing import List


class ObjectBuilder:
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

    def build_from_kwargs(self, obj, **kwargs):
        required_parameters = self._get_required_parameters(obj)
        provided_parameters = kwargs.keys()

        required_set = set(required_parameters)
        provided_set = set(provided_parameters)

        intersection = required_set.intersection(provided_set)
        try:
            if len(intersection) != len(required_set):
                error = "Not enough parameters for {} provided in kwargs! ".format(obj.__name__)
                message = "Required parameters are: {}".format(required_parameters)
                logging.error(error + message)
                raise Exception(error + message)
            else:
                parameters = {parameter: kwargs[parameter] for parameter in required_parameters}
                obj(**parameters)
        except Exception as e:
            error = "Can't build object! {}".format(e)
            logging.error(error)
            raise Exception(error)

    def build_from_map(self, of_class, objects_map, **kwargs):
        if not objects_map.get(of_class):
            error = "Object of class `{}` is not provided in `objects_map`".format(of_class)
            logging.error(error)
            raise TypeError(error)
        else:
            obj = objects_map[of_class]
        return self.build_from_kwargs(name=of_class, obj=obj, **kwargs)
