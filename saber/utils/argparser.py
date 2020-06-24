import json
import inspect


class ArgumentParser(dict):
    __parse_style0__ = "\"{}\":   {}  "
    __parse_style1__ = "\"{}\": \"{}\""

    def __init__(self, *args, caller=None, all_args=None, defaults=None, key_abbrs=None):
        # check requests
        essential_numbers = 0
        essential_args_given = dict()
        # if caller is given
        if caller is not None:
            assert inspect.isclass(caller) or inspect.isfunction(caller),\
                "caller should be class or function, but '{}'".format(type(caller))
            assert all_args is None, "caller is given, all_args should be None"
            assert defaults is None, "caller is given, defaults should be None"
            # get all_args and defaults
            if inspect.isclass(caller):
                all_args = inspect.getargspec(caller.__init__).args[1:]
                defaults = inspect.getargspec(caller.__init__).defaults
            elif inspect.isfunction(caller):
                all_args = inspect.getargspec(caller).args
                defaults = inspect.getargspec(caller).defaults
        # no caller
        if all_args is None:
            assert defaults is None
        else:
            # defaults should be given from the first arg with default value to the last arg
            defaults = defaults or list()
            assert isinstance(defaults, (list, tuple))
            assert isinstance(all_args, (list, tuple))
            assert len(defaults) <= len(all_args)
            essential_numbers = len(all_args) - len(defaults)
            essential_args_given = {key: 0 for key in all_args[:essential_numbers]}
            # first `essential_numbers` args should be given
        key_abbrs = key_abbrs or dict()
        assert isinstance(key_abbrs, dict)
        self.__all_args = all_args
        self.__defaults = defaults
        # parsing
        pos = 0
        self.pos_args = []
        for arg in args:
            if isinstance(arg, str) and arg.find("=") >= 0:
                # named arg
                key, val = self.parse_named_arg(arg)
                key = key_abbrs.get(key, key)
                assert key not in self, "'{}' is duplicated!".format(key)
                super().__setitem__(key, val)
                if key in essential_args_given:
                    essential_args_given[key] = 1
                pos = -1  # no more position args following
            else:
                # position arg
                if pos < 0:
                    raise Exception("position arg '{}' after named args".format(arg))
                self.pos_args.append(arg)
                if all_args is not None and pos < len(all_args):
                    super().__setitem__(all_args[pos], arg)
                    if all_args[pos] in essential_args_given:
                        essential_args_given[all_args[pos]] = 1
                pos += 1
        assert essential_numbers == sum(essential_args_given.values()),\
            "following keys are required but not given: {}".format([
                key for key, count in essential_args_given.items() if count == 0
            ])
        # set defaults is given
        if all_args is not None and defaults is not None:
            for i, default in enumerate(defaults):
                key = all_args[i + essential_numbers]
                if key not in self:
                    super().__setitem__(key, default)

    def __getattr__(self, attr):
        return self.__getitem__(attr)

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        elif isinstance(key, int) and 0 <= key < len(self.pos_args):
            return self.pos_args[key]

    def __repr__(self):
        def _str_pos_arg(arg):
            return (
                "\"{}\"".format(arg)
                if arg in [True, False, None] else
                json.dumps(arg)
            )

        pos_str = ", ".join(_str_pos_arg(x) for x in self.pos_args)
        key_str = ", ".join("\"{}={}\"".format(key, val) for key, val in self.items())
        if self.__all_args is None:
            # no args hints
            # return json.dumps(self.pos_args) + " " + json.dumps(self)
            if len(self.pos_args) > 0 and len(self) > 0:
                return "[{}, {}]".format(pos_str, key_str)
            elif len(self.pos_args) > 0:
                return "[{}]".format(pos_str)
            elif len(self) > 0:
                return "[{}]".format(key_str)
            else:
                return "{}"
        else:
            # with args requiremnets
            if len(self) > 0:
                return "[{}]".format(key_str)
            else:
                return "{}"

    @classmethod
    def parse_pos_arg(cls, arg):
        if isinstance(arg, str):
            if   arg in ["True",  "true"]:  arg = True
            elif arg in ["False", "false"]: arg = False
            elif arg in ["None",  "null"]:  arg = None
        return arg

    @classmethod
    def parse_named_arg(cls, arg):
        splited = arg.split("=")
        assert len(splited) == 2, "named arg should be: <key>=<val> and no '=' in <val>"
        key, val = splited[0], "=".join(splited[1:])
        # parse val
        val = val.replace("'", "\"")  # replace ' to "
        # turn python style to json style
        if   val in ["True",  "true"]:  val = "true"
        elif val in ["False", "false"]: val = "false"
        elif val in ["None",  "null"]:  val = "null"
        try:
            to_parse = cls.__parse_style0__.format(key, val)
            parsed = json.loads("{" + to_parse + "}")
            return key, parsed[key]
        except ValueError:
            pass
        try:
            to_parse = cls.__parse_style1__.format(key, val)
            parsed = json.loads("{" + to_parse + "}")
            return key, parsed[key]
        except ValueError:
            raise ValueError("Failed to parse: '{}'".format(arg))


if __name__ == "__main__":
    parser = ArgumentParser(
        "test", dict(hello=1, haha=[123, 3123]), "linear", True, [12, 3123],
        all_args=["a", "b", "activation"],
        defaults=[None],
        key_abbrs={"act": "activation"}
    )
    print(parser[0])
    print(parser[3], type(parser[3]))
    print(parser[4], type(parser[4]))
    print(parser.activation)
    print(parser)

    print(ArgumentParser(*json.loads(str(parser))))
    print(ArgumentParser("test", {"fuck": 1, "haha": [123, 3123]}, "1", "True", [12, 3123]))
