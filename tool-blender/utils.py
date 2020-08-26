import os
import sys
import time
import argparse
import numpy as np


class timeit(object):
    def __init__(self, tag="timeit"):
        self.tag = tag

    def __enter__(self):
        self.ts = time.time()

    def __exit__(self, *args):
        self.te = time.time()
        print('<{}> cost {:.2f} ms'.format(
            self.tag, (self.te - self.ts) * 1000))
        return False

    def __call__(self, method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            print('<{}> cost {:.2f} ms'.format(
                method.__name__, (te - ts) * 1000))
            return result
        return timed


def read_obj(filename_obj, num_verts):
    # load vertices
    with open(filename_obj) as f:
        lines = f.readlines()

    vi = 0
    vertices = np.zeros((num_verts, 3), np.float32)
    for line in lines:
        line = line.strip().split()
        if len(line) == 0:
            continue
        if line[0] == 'v':
            vertices[vi, 0] = float(line[1])
            vertices[vi, 1] = float(line[2])
            vertices[vi, 2] = float(line[3])
            vi += 1
    return vertices


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())
