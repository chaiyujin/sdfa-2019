import os
import re
from typing import List
from . import log


def ancestor(path, level=1):
    assert level >= 1
    ret = path
    for _ in range(level):
        ret = os.path.dirname(ret)
    return ret


def find_files(directory, pattern, recursive=True, abspath=False):
    regex = re.compile(pattern)
    file_list = []
    for root, _, files in os.walk(directory):
        for f in files:
            if regex.match(f) is not None:
                file_list.append(os.path.join(root, f))
        if not recursive:
            break
    map_func = os.path.abspath if abspath else os.path.relpath
    return list(map(map_func, sorted(file_list)))


def find_dirs(directory, pattern, recursive=True, abspath=False):
    regex = re.compile(pattern)
    dir_list = []
    for root, subdirs, _ in os.walk(directory):
        for f in subdirs:
            if regex.match(f) is not None:
                dir_list.append(os.path.join(root, f))
        if not recursive:
            break
    map_func = os.path.abspath if abspath else os.path.relpath
    return list(map(map_func, sorted(dir_list)))


def maybe_in_dirs(filename, possible_roots=None, possible_exts=None, must_be_found=False):
    """ return first existing filepath according to filename, possible roots and extensions
    """

    def _find_exts(path):
        if os.path.exists(path):
            return path
        if possible_exts is not None:
            assert isinstance(possible_exts, (list, tuple))
            for ext in possible_exts:
                if ext[0] != ".":
                    ext = "." + ext
                path = os.path.splitext(path)[0] + ext
                if os.path.exists(path):
                    return path
        return None

    assert (possible_roots is None) or isinstance(possible_roots, (list, tuple)),\
        "'possible_roots' should be list or tuple. not {}".format(possible_roots)
    assert (possible_exts  is None) or isinstance(possible_exts,  (list, tuple)),\
        "'possible_exts' should be list or tuple. not {}".format(possible_exts)

    fpath = _find_exts(filename)
    if fpath is None:
        for root in possible_roots:
            if root is None or not os.path.exists(root):
                continue
            fpath = _find_exts(os.path.join(root, filename))
            if fpath is not None:
                # log.info("Find file at: {}".format(fpath))
                return fpath
        # log.fatal("Failed to find file: {}".format(filename))
    if must_be_found and fpath is None:
        log.fatal(f"Failed to find file: {filename}")
    return fpath


def maybe_remove_end_separator(path):
    if len(path) > 0 and path[-1] in ["/", "\\"]:
        return path[:-1]
    else:
        return path
