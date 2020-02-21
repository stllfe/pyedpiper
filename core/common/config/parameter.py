from core.common.config.mixins import (
    WrappedPrimitive,
    IsRequiredMixin,
    IsSetMixin,
)


class Parameter(WrappedPrimitive, IsRequiredMixin, IsSetMixin, object):
    def __init__(self, value=None, required=False, *args, **kwargs):
        if isinstance(value, type(self)):
            required, value = value.is_required, value.value
        super(Parameter, self).__init__(value=value, required=required, *args, **kwargs)

    def __repr__(self):
        base_repr = super(Parameter, self).__repr__()
        if self.is_required:
            base_repr += " [required]"
        return base_repr
