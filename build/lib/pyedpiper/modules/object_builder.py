import inspect
import _logging
from typing import List, Dict, Union

log = _logging.getLogger(__name__)


class ObjectBuilder:

    @classmethod
    def _get_parameters(cls, obj_type, ignored_parameters=('self', 'kwargs', 'args')) -> Dict[str, Union[List, Dict]]:
        required_parameters = list()
        optional_parameters = dict()

        signature = inspect.signature(obj_type.__init__)

        def is_required(param):
            return param.default is inspect.Parameter.empty

        for name, value in signature.parameters.items():
            if name in ignored_parameters:
                continue

            if is_required(value):
                required_parameters.append(name)
                continue

            optional_parameters[name] = value.default

        return {'required': required_parameters, 'optional': optional_parameters}

    @classmethod
    def build_from_kwargs(cls, obj_type, name=None, *args, **kwargs):
        assert isinstance(obj_type, (type, object)), "Object of `type` expected, got `{}` instead".format(
            type(obj_type))
        name = name if name else obj_type.__name__
        parameters = cls._get_parameters(obj_type)

        required_parameters = parameters['required']
        optional_parameters = parameters['optional']
        provided_parameters = kwargs.keys()

        required_set = set(required_parameters)
        provided_set = set(provided_parameters)
        optional_set = set(optional_parameters)

        intersection = required_set.intersection(provided_set)
        try:
            if len(intersection) != len(required_set):
                error = ("Not enough parameters! "
                         "Required parameters are: {}".format(', '.join(required_parameters)))
                raise Exception(error)
            else:
                overrides = optional_parameters.copy()
                overrides.update(kwargs)
                signature = optional_set.union(required_set)
                ingredients = {parameter: overrides[parameter] for parameter in signature}
                return obj_type(*args, **ingredients)
        except Exception as e:
            error = "Can't build '{}': {}".format(name, e)
            log.error(error)
            raise e

    @classmethod
    def build_from_map(cls, of_class, objects_map, **kwargs):
        if not objects_map.get(of_class):
            error = "Object of class '{}' is not provided in 'objects_map'".format(of_class)
            log.error(error)
            raise ValueError(error)
        else:
            obj = objects_map[of_class]
        return cls.build_from_kwargs(name=of_class, obj_type=obj, **kwargs)
