import os
import re
import json
import torch
import datetime
import argparse
import numpy as np

# -------------
# ConfigDict object
# -------------
class ConfigDict(dict):

    # for key parsing
    __entirety_key = "__entirety__"
    __ignore_regex = re.compile("^__.+__$")
    __kept_dash_keys = [__entirety_key]

    # handling error
    __err_no_file  = "[ConfigDict]: Failed to find '{}'."
    __err_bad_file = "[ConfigDict]: Given '{}' is not '.json' or '.py'."
    __err_missing  = "[ConfigDict]: Key '{}' is missing."
    __err_constant = "[ConfigDict]: Cannot set constant key '{}'."
    __err_bad_type = "[ConfigDict]: Unknown type '{}'."
    __err_bad_var  = "[ConfigDict]: Unknown var name '{}'."

    # usage message
    __usage        = "[ConfigDict]: Usage: \n"\
                     "  1. ConfigDict(jsonfile:str)\n"\
                     "  2. ConfigDict(dict:dict)\n"\
                     "  3. ConfigDict(**kwargs)"

    def __init__(self, *argv, **kwargs):
        """
        Most keys start and end with "__" will be ignored, such as "__comment__".
        The key "__entirety__" tells that this ConfigDict is treated as an entirety.
        """
        super().__init__()
        if len(argv) == 0 and len(kwargs) == 0:
            pass
        else:
            # check input: only positional or named
            init_from_name = (len(kwargs) >= 1) and (len(argv) == 0)
            init_from_dict = (len(kwargs) == 0) and (len(argv) == 1)
            if not (init_from_name ^ init_from_dict):
                raise ValueError(self.__usage)
            args = argv[0] if init_from_dict else kwargs
            # convert into dict
            if isinstance(args, str):
                # if 'dict' is str, which means a file
                args = self.__dict_from_file(args)
            elif isinstance(args, argparse.Namespace):
                args = vars(args)
            # now, args must be dict
            if not isinstance(args, dict):
                raise TypeError(self.__err_bad_type.format(type(args)))
            # set (key, val) paris
            for key in args:
                if self.__should_ignore(key):
                    continue
                # not ingnored
                value = args[key]
                self.set_key(key, value)
        # default entirety is False
        self.set_missing(self.__entirety_key, False)

    @classmethod
    def __should_ignore(cls, key) -> bool:
        return isinstance(key, str)\
            and (cls.__ignore_regex.match(key) is not None)\
            and (key not in cls.__kept_dash_keys)

    @classmethod
    def __convert_dicts(cls, value) -> object:
        if type(value) is dict:
            value = ConfigDict(value)
        elif isinstance(value, (list, tuple)):
            value = tuple(map(lambda x: cls.__convert_dicts(x), value))
        return value

    @staticmethod
    def date_string():
        date_str = str(datetime.datetime.now())
        date_str = date_str[:date_str.rfind(":")].replace("-", "").replace(":", "").replace(" ", "-")
        return date_str

    # keys =====================================================
    def set_key(self, key, value) -> None:
        """ set key of config dict """
        if key in self.__kept_dash_keys:
            # stored as attr
            super().__setattr__(key, value)
        elif not self.__should_ignore(key):
            # stored as item
            super().__setitem__(key, self.__convert_dicts(value))

    def set_missing(self, key, value) -> None:
        """ set missing key """
        if (key not in self) and (key not in self.__dict__):
            self.set_key(key, value)

    def __deepcopy__(self, memo):
        from copy import deepcopy
        ret = ConfigDict()
        for k in self:
            ret.set_key(k, deepcopy(self[k]))
        for k in self.__dict__:
            ret.set_key(k, deepcopy(self.__dict__[k]))
        return ret

    def __setattr__(self, attr, value):
        raise Exception(self.__err_constant.format(attr))

    def __setitem__(self, item, value):
        raise Exception(self.__err_constant.format(item))

    def get_default(self, key, default=None):
        if key in self:
            return self[key]
        elif key in self.__dict__:
            return self.__dict__[key]
        else:
            return default

    def __getattr__(self, attr):
        if attr in self.__kept_dash_keys:
            return super().__getattr__(attr)
        else:
            return super().__getitem__(attr)

    def check_keys(self, *key_list) -> bool:
        for k in key_list:
            assert k in self, self.__err_missing.format(k)
        return True
    # ==========================================================

    # overwrite ================================================
    def overwrite_by(self, peer) -> None:
        """ overwrite config dict by another peer """
        peer = ConfigDict(peer)
        # check is entirety or not
        if self.__entirety__ or peer.__entirety__:
            # clear self
            self.clear()
            # copy from peer
            for k in peer:
                self.set_key(k, peer[k])
            # should be entirety
            self.set_key(self.__entirety_key, True)
            return
        else:
            # else overwrite keys
            for key in peer:
                assert not self.__should_ignore(key)
                # not continue
                val = peer[key]
                if (key in self) and isinstance(self[key], ConfigDict) and isinstance(val, ConfigDict):
                    self[key].overwrite_by(val)
                else:
                    self.set_key(key, val)
            # should not be entirety
            self.set_key(self.__entirety_key, False)
    # ==========================================================

    # replace ==================================================
    def replace_variable(self, var_name, new_val):
        """ replace {var_name} to new_val """
        var_name = var_name.strip()
        assert var_name[0] == "{" and var_name[-1] == "}", "var_name should be '{name}'"
        assert type(new_val) in [str, int, float, bool], "type of new_val {} is not support!".format(type(new_val))

        def _maybe_replace(old_val):
            if isinstance(old_val, str) and old_val.find(var_name) >= 0:
                # need to replace
                if isinstance(new_val, str):
                    return old_val.replace(var_name, new_val)
                elif isinstance(new_val, (int, float, bool)):
                    assert old_val.strip() == var_name
                    return new_val
                else:
                    raise NotImplementedError()
            else:
                # keep old value
                return old_val

        def _replace_tuple(tup):
            return tuple(
                _replace_tuple(x) if isinstance(x, (tuple, list)) else _maybe_replace(x)
                for x in tup
            )

        for k in self:
            val = self[k]
            if isinstance(val, str):
                self.set_key(k, _maybe_replace(val))
            elif isinstance(val, dict):
                val.replace_variable(var_name, new_val)
            elif isinstance(val, (list, tuple)):
                self.set_key(k, _replace_tuple(val))
    # ==========================================================

    # io =======================================================
    @classmethod
    def __dict_from_file(cls, filename) -> dict:
        # parser json file
        ext = os.path.splitext(filename)[1]
        assert os.path.exists(filename), cls.__err_no_file.format(filename)
        assert ext in [".json", ".py"], cls.__err_bad_file.format(filename)
        if ext == ".json":
            with open(filename, "r") as fp:
                args = json.load(fp)
        elif ext == ".py":
            import importlib.util
            spec = importlib.util.spec_from_file_location("_load_py_hparams", filename)
            hp = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hp)
            assert hasattr(hp, "hparams"), "{} doesn't have hparams!"
            args = hp.hparams
        return args

    def dump(self, filename=None, dump_dir=None, dump_ext=None):
        assert dump_ext in [None, '.json', '.py']
        if filename is None:
            filename = ConfigDict.date_string()
            if dump_ext is not None:
                filename += dump_ext
            else:
                filename += ".json"
        if dump_ext is not None:
            # check extension
            assert os.path.splitext(filename)[1] == dump_ext,\
                "filename '{}' mismatch with dump_ext {}".format(dump_ext)
        dump_ext = os.path.splitext(filename)[1]
        if dump_dir is not None:
            filename = os.path.join(dump_dir, filename)
        if dump_ext == ".json":
            with open(filename, "w") as fout:
                fout.write(str(self))
        else:
            raise NotImplementedError("Dump config into '.py' is not implemented yet!")
    # ==========================================================

    # str ======================================================
    def __str__(self):
        return json.dumps({
            **self,
            "__entirety__": self.__entirety__
        }, cls=MyJsonEncoder, indent=2)
    # ==========================================================


class MyJsonEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        from copy import deepcopy
        self.kwargs = deepcopy(dict(kwargs))
        if "max_width" in kwargs:
            del kwargs["max_width"]
        super(MyJsonEncoder, self).__init__(*args, **kwargs)
        self._replacement_map = {}

    def default(self, o):
        if torch.is_tensor(o):
            o = o.detach().cpu().numpy()
        if isinstance(o, np.ndarray):
            def convert_ndarray(arr):
                if arr.ndim == 1:
                    return [float(x) for x in arr]
                else:
                    return [convert_ndarray(x) for x in arr]
            return convert_ndarray(o)
        return super().default(o)

    def iterencode(self, o, _one_shot=False):
        """ Copy from json.encoder
            change '_iterencode_list'
        """
        from json.encoder import (
            encode_basestring,
            encode_basestring_ascii,
            INFINITY
        )

        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring

        def floatstr(o, allow_nan=self.allow_nan, _repr=float.__repr__, _inf=INFINITY, _neginf=-INFINITY):
            # Check for specials.  Note that this type of test is processor
            # and/or platform-specific, so do tests which don't depend on the
            # internals.

            if o != o:
                text = 'NaN'
            elif o == _inf:
                text = 'Infinity'
            elif o == _neginf:
                text = '-Infinity'
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " +
                    repr(o))

            return text

        max_width = self.kwargs["max_width"] if "max_width" in self.kwargs else 120
        _iterencode = _make_iterencode(
            markers, self.default, _encoder, self.indent, floatstr,
            self.key_separator, self.item_separator, self.sort_keys,
            self.skipkeys, _one_shot, max_width
        )
        return _iterencode(o, 0)


def _make_iterencode(
    markers, _default, _encoder, _indent, _floatstr,
    _key_separator, _item_separator, _sort_keys, _skipkeys, _one_shot, max_width,
    # HACK: hand-optimized bytecode; turn globals into locals
    ValueError=ValueError,
    dict=dict,
    float=float,
    id=id,
    int=int,
    isinstance=isinstance,
    list=list,
    str=str,
    tuple=tuple,
    _intstr=int.__str__,
):

    if _indent is not None and not isinstance(_indent, str):
        _indent = ' ' * _indent

    def _iterencode_list(lst, _current_indent_level, key_width=0):
        if not lst:
            yield '[]'
            return
        if markers is not None:
            markerid = id(lst)
            if markerid in markers:
                raise ValueError("Circular reference detected")
            markers[markerid] = lst
        buf = '['
        if _indent is not None:
            _current_indent_level += 1
            newline_indent = '\n' + _indent * _current_indent_level
            buf += newline_indent
        else:
            newline_indent = ''
        # first convert in to children
        children = []
        for value in lst:
            if isinstance(value, str):
                children.append(_encoder(value))
            elif value is None:
                children.append('null')
            elif value is True:
                children.append('true')
            elif value is False:
                children.append('false')
            elif isinstance(value, int):
                # Subclasses of int/float may override __str__, but we still
                # want to encode them as integers/floats in JSON. One example
                # within the standard library is IntEnum.
                children.append(_intstr(value))
            elif isinstance(value, float):
                # see comment above for int
                children.append(_floatstr(value))
            else:
                if isinstance(value, (list, tuple)):
                    chunks = _iterencode_list(value, _current_indent_level)
                elif isinstance(value, dict):
                    chunks = _iterencode_dict(value, _current_indent_level)
                else:
                    chunks = _iterencode(value, _current_indent_level)
                children.append(''.join(chunks))
        try_one = '[' + _item_separator.join(children) + ']'
        indent_width = len(_indent * _current_indent_level)
        if len(try_one) <= max_width - indent_width - key_width:
            yield try_one
        else:
            one_line = ""
            for ci, child in enumerate(children):
                # a new line
                if len(one_line) == 0:
                    if ci == 0:
                        yield buf
                else:
                    one_line += _item_separator
                new_line = one_line + child
                if len(new_line) < max_width - indent_width:
                    one_line = new_line
                else:
                    yield one_line + newline_indent
                    one_line = child
            yield one_line
            if len(newline_indent) > 0:
                _current_indent_level -= 1
                yield '\n' + _indent * _current_indent_level
            yield ']'
        if markers is not None:
            del markers[markerid]

    def _iterencode_dict(dct, _current_indent_level):
        if not dct:
            yield '{}'
            return
        if markers is not None:
            markerid = id(dct)
            if markerid in markers:
                raise ValueError("Circular reference detected")
            markers[markerid] = dct
        yield '{'
        if _indent is not None:
            _current_indent_level += 1
            newline_indent = '\n' + _indent * _current_indent_level
            item_separator = _item_separator + newline_indent
            yield newline_indent
        else:
            newline_indent = None
            item_separator = _item_separator
        first = True
        if _sort_keys:
            items = sorted(dct.items(), key=lambda kv: kv[0])
        else:
            items = dct.items()
        for key, value in items:
            if isinstance(key, str):
                pass
            # JavaScript is weakly typed for these, so it makes sense to
            # also allow them.  Many encoders seem to do something like this.
            elif isinstance(key, float):
                # see comment for int/float in _make_iterencode
                key = _floatstr(key)
            elif key is True:
                key = 'true'
            elif key is False:
                key = 'false'
            elif key is None:
                key = 'null'
            elif isinstance(key, int):
                # see comment for int/float in _make_iterencode
                key = _intstr(key)
            elif _skipkeys:
                continue
            else:
                raise TypeError("key " + repr(key) + " is not a string")
            if first:
                first = False
            else:
                yield item_separator
            yield _encoder(key)
            yield _key_separator
            if isinstance(value, str):
                yield _encoder(value)
            elif value is None:
                yield 'null'
            elif value is True:
                yield 'true'
            elif value is False:
                yield 'false'
            elif isinstance(value, int):
                # see comment for int/float in _make_iterencode
                yield _intstr(value)
            elif isinstance(value, float):
                # see comment for int/float in _make_iterencode
                yield _floatstr(value)
            else:
                if isinstance(value, (list, tuple)):
                    chunks = _iterencode_list(value, _current_indent_level, key_width=len(_encoder(key)+_key_separator))
                elif isinstance(value, dict):
                    chunks = _iterencode_dict(value, _current_indent_level)
                else:
                    chunks = _iterencode(value, _current_indent_level)
                yield from chunks
        if newline_indent is not None:
            _current_indent_level -= 1
            yield '\n' + _indent * _current_indent_level
        yield '}'
        if markers is not None:
            del markers[markerid]

    def _iterencode(o, _current_indent_level):
        if isinstance(o, str):
            yield _encoder(o)
        elif o is None:
            yield 'null'
        elif o is True:
            yield 'true'
        elif o is False:
            yield 'false'
        elif isinstance(o, int):
            # see comment for int/float in _make_iterencode
            yield _intstr(o)
        elif isinstance(o, float):
            # see comment for int/float in _make_iterencode
            yield _floatstr(o)
        elif isinstance(o, (list, tuple)):
            yield from _iterencode_list(o, _current_indent_level)
        elif isinstance(o, dict):
            yield from _iterencode_dict(o, _current_indent_level)
        else:
            if markers is not None:
                markerid = id(o)
                if markerid in markers:
                    raise ValueError("Circular reference detected")
                markers[markerid] = o
            o = _default(o)
            yield from _iterencode(o, _current_indent_level)
            if markers is not None:
                del markers[markerid]
    return _iterencode


if __name__ == "__main__":

    print(ConfigDict.date_string())

    config = ConfigDict(
        dataset=dict(
            emotions=dict(
                neutral     = 0,
                happy       = 1,
                surprised   = 2,
                angry       = 3,
                sad         = 4,
                __entirety__= True
            ),
        ),
        names=["haha", "caonima"],
        matrix=np.asarray([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]),
        empty=list(range(38))+[0, 1]
    )

    config.overwrite_by("../../test.py")
    config.overwrite_by("../../test1.py")

    print(config)
    # print(config.dataset.emotions)
    # print(config.dataset.emotions.__dict__)
    # config.dump("config.json")
