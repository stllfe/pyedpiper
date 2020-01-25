import inspect

from pathlib import Path
from importlib import import_module
from importlib.util import spec_from_file_location, module_from_spec

from src.common.consts import (
    MODELS_DIR,
    SCHEDULERS,
    OPTIMIZERS,
    METRICS,
    LOSSES,
)


class ModelObjectProvider:
    def __init__(self, model_name, load_from):
        self.model_name = model_name
        self.load_from = load_from
        self.loaded_from = None
        self.model_obj = self._load_model()

    @staticmethod
    def _load_third_party_module(self, module_name):
        try:
            return import_module(module_name)
        except ModuleNotFoundError:
            return

    @staticmethod
    def _load_local_module(self, module_path):
        module_path = Path(module_path)
        name = module_path.name.split('.')[0]
        if not module_path.exists():
            return
        spec = spec_from_file_location(
            name=name,
            location=module_path
        )
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_model(self):
        load_from = self.load_from
        if load_from:
            print("loading module `%s`" % load_from)
            module = self._load_third_party_module(self, load_from)
        else:
            # if no module provided search locally
            print("searching in `%s` directory ..." % MODELS_DIR)
            load_from = MODELS_DIR / ('%s.py' % self.model_name.lower())
            module = self._load_local_module(self, load_from)

        if not module:
            error = "error! module not found: `%s`" % load_from
            raise ModuleNotFoundError(error)

        print("found module with name: %s " % module.__name__)
        try:
            self.model_obj = getattr(module, self.model_name)
        except AttributeError as e:
            error = "error! no object `%s` found in module: `%s`" % (self.model_name, load_from)
            print(error)
            raise e

        # save the final destination if successfully loaded
        self.loaded_from = load_from
        print("model `%s` loaded successfully" % self.model_name)

    def get_model(self):
        pass


class PipelineObjectsBuilder:
    def __init__(self, config, model_provider):
        self.config = config
        model_provider = model_provider(model_name=config.model,
                                        load_from=config.load_from)
        self.model = model_provider.get_model()
        self.engine = None

    def _build_obj_from_config(self, name, obj, ignored_parameters=('self',), **kwargs):
        obj_type = self.config[name].lower()
        signature = inspect.signature(obj.__init__)

        def is_required(value):
            return value.default is inspect.Parameter.empty

        parameters = dict(**kwargs)
        for parameter, value in signature.parameters.items():
            if parameter not in parameters.keys() and parameter not in ignored_parameters:
                if self.config[parameter]:
                    parameters[parameter] = self.config[parameter]
                elif is_required(value):
                    error = "error! can't find required parameter `%s` for `%s` type `%s`:"
                    message = "please update your config files."
                    raise Exception(error % (parameter, name, obj_type), message)

        try:
            return obj(**parameters)
        except Exception:
            error = "error! can't build object: `%s`, not enough parameters provided in kwargs" % name
            raise Exception(error)

    def _build(self, name, available_objects, **kwargs):
        obj_type = self.config[name].lower()
        default = available_objects.get('default')

        if not default:
            # means this thing is optional
            return

        if not obj_type:
            obj = default

        elif not available_objects.get(obj_type):
            error = "error! %s type `%s` is not implemented" % (name, obj_type)
            raise NotImplementedError(error)

        else:
            obj = available_objects[obj_type]

        return self._build_obj_from_config(name, obj=obj, **kwargs)


class TrainObjectsBuilder(PipelineObjectsBuilder):
    def __init__(self, config, model_loader):
        super().__init__(config, model_loader)
        self.optimizer = self._build('optimizer', OPTIMIZERS, params=self.model.parameters())
        self.scheduler = self._build('scheduler', SCHEDULERS, optimizer=self.optimizer)
        self.loss_fn = self._build('loss', LOSSES[config.task_type])


class TestObjectsBuilder(PipelineObjectsBuilder):
    pass


import torch

loss = torch.nn.SmoothL1Loss()
