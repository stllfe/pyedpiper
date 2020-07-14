import torch


def concat_outputs(outputs: list, prefix="") -> dict:
    """

    :param outputs: list of outputs from series of PyTorch Lightning steps.
                    You get it as an argument for '...__epoch_end()' methods
    :param prefix: prefix to add for every key in averaged dictionary. default - empty string
    :return: dict with every tensor value concatenated across all outputs
    """

    def _recursive_concat(outputs):
        concat = dict()
        if not outputs:
            return concat

        if len(outputs) == 1:
            return outputs[0]

        # We need only one pass since every dict in list contains the same metrics, hence use only the first item
        output = outputs[0]
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                # todo: calculate output shape explicitly
                stacked = torch.stack([output[key] for output in outputs])
                flatten = stacked.flatten(end_dim=-2) if stacked.ndim > 2 else stacked.flatten()
                new_key = (prefix + '_' + key).strip('_')
                concat[new_key] = flatten
                continue
            elif isinstance(value, dict):
                new_dict = _recursive_concat([output[key] for output in outputs])
                concat[key] = new_dict
                continue
            else:
                # log.warning('Non-typical value detected in outputs. Are you sure you use this function correctly?')
                concat[key] = value
        return concat

    return _recursive_concat(outputs)


def average_metrics(outputs: list, prefix="") -> dict:
    """
    Averages all the outputs, adds 'avg' prefix and inserts
    :param outputs: list of outputs from series of PyTorch Lightning steps.
                    You get it as an argument for '__epoch_end' methods
    :param prefix: prefix to add for every key in averaged dictionary. default - empty string
    :return: dict with every tensor value averaged across all outputs
    """
    # todo: simplify it to `concat_metrics` -> mean
    def _recursive_average(outputs):
        averages = dict()
        if not outputs:
            return averages

        # We need only one pass since every dict in list contains the same metrics, hence use only the first item
        output = outputs[0]
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                average = torch.stack([output[key] for output in outputs]).mean()
                new_key = (prefix + '_' + key).strip('_')
                averages[new_key] = average
                continue
            elif isinstance(value, dict):
                new_dict = _recursive_average([output[key] for output in outputs])
                averages[key] = new_dict
                continue
            else:
                # log.warning('Non-typical value detected in outputs. Are you sure you use this function correctly?')
                averages[key] = value
        return averages

    return _recursive_average(outputs)


def map_outputs_to_label(output: dict, old_label, new_label) -> dict:
    """
    Shorthand to swap outputs labels to a different one. For example swap 'val' metrics prefix to 'test' or vice versa.
    :param output: dict with outputs from PyTorch Lightning step such as 'val_step' or 'test_step'
    :param old_label: str
    :param new_label: str
    :return:
    """

    def _recursive_map(output):
        new_outputs = dict()
        for key, value in output.items():
            new_key = key
            if isinstance(value, dict):
                value = _recursive_map(value)
            elif isinstance(value, torch.Tensor):
                new_key = key.replace(old_label, new_label)
            new_outputs[new_key] = value
        return new_outputs

    return _recursive_map(output)


def extract_unique_metrics(results: dict) -> dict:
    """
    Extract all metrics from dict or nested dicts recursively.
    According to PyTorch Lightning API output metrics are of type '_torch.Tensors'.
    :param results: dict from any step output
    :return: flat dict with all the metrics found
    """

    def _extract_recursive(results):
        metrics = dict()
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value
            elif isinstance(value, dict):
                metrics.update(_extract_recursive(value))
        return metrics

    return _extract_recursive(results)


def average_and_log_metrics(outputs: list, prefix=""):
    """
    Shorthand for '$step_epoch_end' type of steps according to PyTorch Lightning API.
    It averages all the outputs, adds prefix if provided and inserts new key 'log' for logger to capture the output.
    :param outputs: outputs from series of PyTorch Lightning steps such as 'validation_epoch_end' inputs or 'test_epoch_end'
    :param prefix: prefix to add for every key in averaged dictionary. default - empty string
    :return: new dict as specified in description
    """
    # todo: add a log filter, to choose specific keys for _logging

    results = average_metrics(outputs, prefix)
    results.update({'log': extract_unique_metrics(results)})
    return results
