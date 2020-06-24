import json
import inspect
import numpy as np
from saber.utils import log
from .conv1d import Conv1d, ConvTranspose1d, Pool1d, Residual1d, ResidualStack1d
from .conv2d import Conv2d, ConvTranspose2d, Pool2d
from .linear import FullyConnected, FeatureProjection
from .reshape import Flatten, Permute, Transpose, Squeeze, Unsqueeze, View
from .extend import ILayerExtended
from ..functions import Identity, GradScaler


class LayerParser(dict):
    __brevs__ = {
        "act":       "activation",
        "ksz":       "kernel_size",
        "hop":       "stride",
        "pad":       "padding",
        "dil":       "dilation",
        "in":        "in_channels",
        "out":       "out_channels",
        "init":      "init_method",
        "nonlinear": "init_nonlinearity"
    }

    __layer_types__ = {
        "conv1d":    Conv1d,
        "conv2d":    Conv2d,
        "deconv1d":  ConvTranspose1d,
        "deconv2d":  ConvTranspose2d,
        "pool1d":    Pool1d,
        "pool2d":    Pool2d,
        "view":      View,
        "flatten":   Flatten,
        "permute":   Permute,
        "transpose": Transpose,
        "squeeze":   Squeeze,
        "unsqueeze": Unsqueeze,
        "res1d":     ResidualStack1d,
        "identity":  Identity,
        "gradx":     GradScaler,
        "fc":        FullyConnected,
        "fp":        FeatureProjection
    }

    def __init__(self, layer_info, creation=None, can_ignore_keys=[]):
        assert isinstance(layer_info, (list, tuple))
        assert len(layer_info) > 0
        # get layer class
        if creation is None:
            assert layer_info[0] in self.__layer_types__
            self.creation = self.__layer_types__[layer_info[0]]
        else:
            self.creation = creation
        # get args information
        if inspect.isclass(self.creation):
            all_args = inspect.getfullargspec(self.creation.__init__).args[1:]
            defaults = inspect.getfullargspec(self.creation.__init__).defaults
        elif inspect.isfunction(self.creation):
            all_args = inspect.getfullargspec(self.creation).args
            defaults = inspect.getfullargspec(self.creation).defaults
        else:
            raise TypeError("given creation is not a class or function, but '{}'".format(type(creation)))
        defaults = defaults or []
        num_pos_args = len(all_args) - len(defaults)
        pos_arg_set = [False for _ in range(num_pos_args)]
        name_index = {name: i for i, name in enumerate(all_args)}
        self.name = layer_info[0]
        self.ignore_args = dict()
        # parse args
        idx = 0
        for arg in layer_info[1:]:
            if isinstance(arg, str) and arg.find("=") > 0:
                # named arg
                arg = arg.split("=")
                name, argstr = arg[0], "=".join(arg[1:])
                # parse arg
                name = self.parse_name(name)
                argstr = argstr.replace("'", "\"")
                if   argstr in ["True", "true"]: argstr = "true"
                elif argstr in ["False", "false"]: argstr = "false"
                elif argstr in ["None", "null"]: argstr = "null"
                try:
                    to_parse = "{" + "\"{}\": {}".format(name, argstr) + "}"
                    parsed = json.loads(to_parse)
                except ValueError:
                    to_parse = "{" + "\"{}\": \"{}\"".format(name, argstr) + "}"
                    parsed = json.loads(to_parse)
                # print("->", name, parsed[name], type(parsed[name]))
                if name in all_args:
                    super().__setitem__(name, parsed[name])
                    if name_index[name] < num_pos_args:
                        pos_arg_set[name_index[name]] = True
                else:
                    super().__setitem__(name, parsed[name])

                idx = -1
            elif idx >= 0:
                # pos arg
                super().__setitem__(all_args[idx], arg)
                if idx < len(pos_arg_set):
                    pos_arg_set[idx] = True
                idx += 1
            else:
                raise ValueError("position arg after named args: '{}'".format(arg))
        if not all(pos_arg_set):
            for flag, name in zip(pos_arg_set, all_args):
                if not flag:
                    log.error("[{}] position arg is not set: '{}'".format(self.name, name))
            quit()

    def __getattr__(self, attr):
        return self.__getitem__(attr)

    def __getitem__(self, item):
        if item in self:
            return super().__getitem__(item)
        elif item in self.ignore_args:
            return self.ignore_args[item]
        else:
            raise KeyError("unknown key '{}'".format(item))

    def get(self, key, default=None):
        if key in self or key in self.ignore_args:
            return self.__getitem__(key)
        else:
            return default

    def create(self, verbose):
        m = self.creation(**self)
        if verbose:
            log.info("create '{}', args: {}".format(self.name, self))
        return m

    @classmethod
    def parse_name(cls, name):
        if name in cls.__brevs__:
            return cls.__brevs__[name]
        else:
            return name

    @staticmethod
    def print_table(title, *parsers):
        rows = []
        cols_width = []
        # collect all meta keys
        meta_keys = []
        for parser in parsers:
            for key in parser:
                if key not in meta_keys:
                    meta_keys.append(key)
        meta_keys = ["name"] + meta_keys
        rows.append(meta_keys)
        for col in rows[0]:
            cols_width.append(len(col))
        # prepare rows
        for parser in parsers:
            this_row = []
            for c, key in enumerate(meta_keys):
                if key == "name":
                    val = parser.name
                elif key == "batch_norm":
                    val = parser.get(key)
                    if isinstance(val, dict):
                        valstr = ""
                        for k in val:
                            if len(valstr) > 0:
                                valstr += ","
                            valstr += f"{k[:3]}={val[k]}"
                        val = valstr
                else:
                    val = str(parser.get(key, "-"))
                if val is None:
                    val = "-"
                cols_width[c] = max(cols_width[c], len(val))
                this_row.append(val)
            rows.append(this_row)
        # generate rows string
        log.info(title)
        for r in range(len(rows)):
            for c in range(len(rows[r])):
                x = rows[r][c]
                pl = (cols_width[c] - len(x)) // 2
                pr = cols_width[c] - len(x) - pl
                rows[r][c] = " "*(pl+1) + x + " "*(pr+1)
            rows[r] = "|" + "|".join(
                x
                for c, x in enumerate(rows[r])
            ) + "|"
            log.println(rows[r], color=None)


# -------------
# easy creation
# -------------
def create(layer_info, verbose=False):
    parser = LayerParser(layer_info)
    return parser.create(verbose), parser


if __name__ == "__main__":
    layer_info = ["conv1d", 16, 64, "ksz=3", "hop=1", "pad=1"]
    parser = LayerParser(layer_info)
    print(parser)
    print(parser.kernel_size)
    print(parser.in_channels)
    m = parser.create(False)
