# -*- coding: utf-8 -*-

import json
from utils.osutils import make_dir


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    make_dir(fpath)
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ':'))