from src.common.modules import ModelObjectProvider
from src.common.modules import TrainObjectsBuilder


class XceptionModel(ModelObjectProvider):
    def get_model(self):
        parameters = {
            'num_classes': self.num_outputs,
            'pretrained': 'imagenet',
        }
        model = self.model_obj(**parameters)
        return model
