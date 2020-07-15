import inspect
import logging

from collections import OrderedDict, deque

log = logging.getLogger(__name__)

_IGNORED_PARAMETERS = ('self', 'kwargs', 'args')


def _is_required(param):
    return param.default is inspect.Parameter.empty


class ObjectCaller:

    @classmethod
    def call_from_kwargs(cls, obj_type, *args, **kwargs, ):
        # Validate input object
        # if inspect.isclass(obj_type):
        #     obj_type = obj_type.__init__

        # if not inspect.isfunction(obj_type) or inspect.isclass(obj_type):
        #     raise TypeError("Can call type or function objects only!")

        assert callable(obj_type), "Can call type or function objects only!"

        name = obj_type.__name__
        sign = inspect.signature(obj_type)

        required_parameters = OrderedDict()
        optional_parameters = OrderedDict()

        # Collect all the necessary params
        for name, value in sign.parameters.items():
            if name in _IGNORED_PARAMETERS:
                continue

            if _is_required(value):
                required_parameters[name] = None
                continue

            optional_parameters[name] = value.default

        # First try to map the args
        if args:
            q = deque(args)
            for name, value in required_parameters.items():
                kwargs[name] = q.popleft()
                if not q:
                    break

        provided_parameters = kwargs.keys()

        required_set = set(required_parameters)
        provided_set = set(provided_parameters)
        optional_set = set(optional_parameters)

        intersection = required_set.intersection(provided_set)
        try:
            diff = len(required_set) - len(intersection)
            if diff:
                error = ("Not enough parameters provided! "
                         "Required parameters are: {}".format(', '.join(required_parameters)))
                raise Exception(error)
            else:
                overrides = optional_parameters.copy()
                overrides.update(kwargs)
                signature = optional_set.union(required_set)
                ingredients = {parameter: overrides[parameter] for parameter in signature}
                return obj_type(**ingredients)
        except Exception as e:
            error = "Can't call `{}`: {}".format(name, e)
            log.error(error)
            raise e

    @classmethod
    def call_from_map(cls, of_class, objects_map, **kwargs):
        if not objects_map.get(of_class):
            error = "Object `{}` is not provided in `objects_map`".format(of_class)
            log.error(error)
            raise ValueError(error)
        else:
            obj = objects_map[of_class]
        return cls.call_from_kwargs(name=of_class, obj_type=obj, **kwargs)
