# -*- coding: utf-8 -*-

import os
import errno

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise ValueError('make dir error!')
