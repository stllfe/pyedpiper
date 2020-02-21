# @on_init(action=fill_defaults)
# class DefaultConfig(Config):
#     @classmethod
#     def for_regression(cls, metric="MAE", data_type="images"):
#         regression_config = cls()
#         regression_config.task_type = TaskTypes.Regression
#         regression_config.num_outputs = 1
#         regression_config.train.metrics.append(metric)
#         regression_config.test.metrics.append(metric)
#
#         if is_implemented_type(data_type, DataTypes):
#             regression_config.data_type = DataTypes[snake_to_camel(data_type.lower())]
#             return regression_config
#
#         regression_config.data_type = DataTypes.Custom
#         return regression_config
#
#     @classmethod
#     def for_classification(cls, metric="CrossEntropyLoss", data_type="images", num_classes=2):
#         classification_config = cls()
#         classification_config.task_type = TaskTypes.Classification
#         classification_config.num_outputs = num_classes
#         classification_config.train.metrics.append(metric)
#         classification_config.test.metrics.append(metric)
#
#         if is_implemented_type(data_type, DataTypes):
#             classification_config.data_type = DataTypes[snake_to_camel(data_type.lower())]
#             return classification_config
#
#         classification_config.data_type = DataTypes.Custom
#         return classification_config
