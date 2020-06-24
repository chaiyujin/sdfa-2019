from .config_dict import ConfigDict
from .decorators import (
    extend,
    extend_classmethod,
    extend_staticmethod
)
from .extension import *
from .bilateral import BilateralFilter1D
from .argparser import ArgumentParser


def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property
