import io
import os
import sys
import time
import inspect
import datetime
from colorama import init
from termcolor import colored
from tqdm import tqdm

init()

__std = [sys.stdout]
__err = [sys.stderr]
__map_std = {}
__map_err = {}


def is_console(fp):
    return fp == sys.stdout or fp == sys.stderr


def time_str():
    return "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "]"


def append_out(file):
    if not isinstance(file, io.TextIOWrapper):
        assert isinstance(file, str)
        dirname = os.path.dirname(file)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
        fp = open(file, "a")
    else:
        fp = file
    if not (fp in __std):
        __map_std[file] = fp
        __std.append(fp)


def append_err(file):
    if not isinstance(file, io.TextIOWrapper):
        assert isinstance(file, str)
        dirname = os.path.dirname(file)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
        fp = open(file, "a")
    else:
        fp = file
    if not (fp in __err):
        __map_err[file] = fp
        __err.append(fp)


def remove(file):
    if file in __map_std:
        fp = __map_std[file]
        if isinstance(file, str):
            fp.close()
        del __map_std[file]
        __std.remove(fp)
    elif file in __map_err:
        fp = __map_err[file]
        if isinstance(file, str):
            fp.close()
        del __map_err[file]
        __err.remove(fp)


def println(*args, color="green", flush=True):
    for fp in __std:
        if is_console(fp) and color is not None:
            args = (colored(arg, color) for arg in args)
        print(*args, file=fp, flush=flush)


def info(*args, flush=True):
    for fp in __std:
        prefix = "[saber][+]:"
        if is_console(fp):
            prefix = colored(prefix, "green")
        print(prefix, *args, file=fp, flush=flush)


def warn(*args, flush=True):
    for fp in __std:
        prefix = "[saber][!]:"
        if is_console(fp):
            prefix = colored(prefix, "yellow")
        print(prefix, *args, file=fp, flush=flush)


def error(*args, flush=True):
    for fp in __err:
        prefix = "[saber][e]:"
        pos    = "  File:"
        func   = "Function:"
        code   = "    -> "
        if is_console(fp):
            prefix = colored(prefix, "red")
            pos = colored(pos, "red")
            func = colored(func, "red")
            code = colored(code, "red")
        print(prefix, *args, file=fp, flush=flush)


def fatal(*args, flush=True):
    caller_list = inspect.getouterframes(inspect.currentframe(), 1)
    for fp in __err:
        prefix = "[saber][f]:"
        pos    = "  File:"
        func   = "Function:"
        code   = "    -> "
        if is_console(fp):
            prefix = colored(prefix, "white", "on_magenta")
            pos = colored(pos, "magenta")
            func = colored(func, "magenta")
            code = colored(code, "magenta")
        print(prefix, *args, file=fp, flush=flush)
        caller_info = caller_list[1]
        if caller_info[4] is not None:
            print(pos, "\""+caller_info[1]+":"+str(caller_info[2])+"\"",
                  func, caller_info[3], file=fp, flush=flush)
            print(code, caller_info[4][0].strip(), file=fp, flush=flush)
    quit()


def assertion(flag, *args, flush=True):
    if not flag:
        caller_list = inspect.getouterframes(inspect.currentframe(), 1)
        for fp in __err:
            prefix = "[saber][a]: trace back"
            detail = "  [detail]:"
            pos    = "  [{:2d}]:"
            code   = "     ->"
            if is_console(fp):
                prefix = colored(prefix, 'white', 'on_magenta')
                detail = colored(detail, 'magenta')
                pos = colored(pos, 'magenta')
                code = colored(code, 'magenta')
            print(prefix, file=fp, flush=flush)
            for li in range(len(caller_list) - 1, 0, -1):
                caller_info = caller_list[li]
                if caller_info[4] is not None:
                    print(
                        pos.format(li-1), "File:",
                        "\""+caller_info[1]+":"+str(caller_info[2])+"\"",
                        "Function:", caller_info[3],
                        file=fp, flush=flush
                    )
                    print(
                        code, caller_info[4][0].strip(),
                        file=fp, flush=flush
                    )
            print(detail, *args, file=fp, flush=flush)
        quit()


# timeit
class timeit(object):
    def __init__(self, tag="timeit"):
        self.tag = tag

    def __enter__(self):
        self.ts = time.time()

    def __exit__(self, *args):
        self.te = time.time()
        info('<{}> cost {:.2f} ms'.format(
            self.tag, (self.te - self.ts) * 1000))
        return False

    def __call__(self, method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            info('<{}> cost {:.2f} ms'.format(
                method.__name__, (te - ts) * 1000))
            return result
        return timed


if __name__ == "__main__":
    append_out("std.log")
    append_err("err.log")
    info("this is normally")
    warn("this function {}".format(__file__))
    error(1, 2, 3)
    a = 1
    b = 1
    assertion(a == b, "{} != {}".format(a, b))
    fatal("fatal")
