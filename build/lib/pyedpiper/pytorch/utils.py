from collections import OrderedDict

from torch.nn import Module

import logging

log = logging.getLogger()


def transfer_weights(model: Module, state_dict: OrderedDict, verbose=False) -> Module:
    """Copy weights from state dict to model, skipping layers that are incompatible.

    This method is helpful if you are doing some model surgery and want to load
    part of the model weights into different model.

    Args:
        model (Module): Model to load weights into
        state_dict (OrderedDict): Model state dict to load weights from
        verbose (bool): whether to print unmatched layers

    Returns:
        Module: The model
    """
    missing_keys = list()
    unexpected_keys = list()
    for name, value in state_dict.items():
        try:
            keys = model.load_state_dict(OrderedDict([(name, value)]), strict=False)
            missing_keys += keys.missing_keys
            unexpected_keys += keys.unexpected_keys
        except Exception as e:
            log.error(f"Error occurred while loading {value} into {name}. \n {e}")
            return model

    if verbose:
        log.info(f"Transfer completed. "
                 f"Unexpected keys: {', '.join(unexpected_keys)}. "
                 f"Missing keys: {', '.join(missing_keys)}.")

    return model
