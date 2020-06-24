import os
import pandas
import numpy as np
from pydoc import locate
from saber import log

_types = ['int', 'str', 'path', 'float']


def _check_meta(meta):
    m = meta.split(":")
    assert len(m) == 2, "meta `{}` is not in format <name>:<type>".format(meta)
    assert m[1] in _types, "<type> should be in {}".format(_types)
    return True


def _get_types(metadata):
    types = []
    for meta in metadata:
        name, type_str = meta.split(":")
        if type_str == "path":
            type_str = "str"
        t = locate(type_str)
        types.append(t)
    return types


def meta_is_path(meta):
    """ metadata <any>:path describes a path """
    _, type_str = meta.split(":")
    return type_str == "path"


def write_csv(metadata, datadicts, output_file, save_relpath=True, spliter=",",
              need_information=False):
    # ignore the empty dicts
    if len(datadicts) == 0:
        return
    # check metadata
    _ = [_check_meta(meta) for meta in metadata]
    # fix extension to '.csv'
    output_file = os.path.splitext(output_file)[0] + ".csv"
    info_file = os.path.splitext(output_file)[0] + ".info"
    # write datadicts
    dirname = os.path.dirname(output_file)
    with open(output_file, "w", encoding='utf-8') as fp:
        fp.write(spliter.join(metadata) + "\n")
        for data in datadicts:
            line = spliter.join([os.path.relpath(str(data[meta]), dirname)
                                 if (meta_is_path(meta) and save_relpath)
                                 else str(data[meta])
                                 for meta in metadata])
            fp.write(line + "\n")
    if not need_information:
        return
    # information
    descriptions = []
    # for 'int' type, generate information
    for meta in metadata:
        name, type_str = meta.split(":")
        if type_str not in ['int', 'float']:
            continue
        tp = locate(type_str)
        vmin = min(tp(d[meta]) for d in datadicts)
        vmax = max(tp(d[meta]) for d in datadicts)
        descriptions.append("{}: {}~{}".format(name, vmin, vmax))
    information = "{} tuples:\n" + "\n".join(descriptions)
    with open(info_file, "w", encoding='utf-8') as fp:
        fp.write(information)
    log.info(information)


def read_csv(csv_path, spliter=","):
    assert os.path.exists(csv_path), "No csv file: '{}'".format(csv_path)
    dirname = os.path.dirname(csv_path)
    metadata = None
    datadicts = []

    df = pandas.read_csv(csv_path, sep=spliter)
    # check metadata
    metadata = df.columns.values
    _ = [_check_meta(m) for m in metadata]
    types = _get_types(metadata)
    # read tuples
    for row in log.tqdm(df.values, desc="read csv"):
        data = {
            meta: (t(d) if not meta_is_path(meta)
                   else os.path.join(dirname, str(d)))
            for d, t, meta in zip(row, types, metadata)
        }
        datadicts.append(data)

    return metadata, datadicts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str)
    args = parser.parse_args()
