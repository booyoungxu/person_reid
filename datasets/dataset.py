# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
from .serialization import read_json
import numpy as np


def filterate(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices): # pid set
        image_pids = identities[pid]
        for cam_id, cam_images in enumerate(image_pids): # each camera's images
            for fname in cam_images:
                name = os.path.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert x == pid and y == cam_id
                if relabel:
                    ret.append((fname, index, cam_id))
                else:
                    ret.append((fname, pid, cam_id))
    return ret


class DataSet(object):
    def __init__(self, root, split_index=0):
        self.root = root
        self.split_index = split_index
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property # jiang fangfa biancheng shuxing diaoyong keyi zhiyou getter er meiyou setter,bianyu canshu jiancha
    def images_dir(self):
        return os.path.join(self.root, 'images')

    def load(self, num_val=100, verbose=True):
        splits = read_json(os.path.join(self.root, 'splits.json'))
        if self.split_index > len(splits):
            raise ValueError('split_index must small than total splits {}'.format(len(splits)))
        self.split = splits[self.split_index]
        trainval_pids = np.asarray(self.split['trainval'])
        np.random.shuffle(trainval_pids)
        num = len(trainval_pids)
        if isinstance(num_val, float):
            num_val = int(round(num*num_val))
        if num_val >= num or num_val < 0:
            raise ValueError('num_val is incorrect')

        train_pids = sorted(trainval_pids[:-num_val])
        val_pids = sorted(trainval_pids[-num_val:])

        self.meta = read_json(os.path.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        self.train = filterate(identities, train_pids, True)
        self.trainval = filterate(identities, trainval_pids, True)
        self.val = filterate(identities, val_pids, True)
        self.query = filterate(identities, self.split['query'])
        self.gallery = filterate(identities, self.split['gallery'])
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)

        if verbose:
            print(self.__class__.__name__, 'loaded')
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  val      | {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.val)))
            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))

    def check_integrity(self):
        return os.path.isdir(os.path.join(self.root, 'images')) and \
               os.path.isfile(os.path.join(self.root, 'meta.json')) and \
               os.path.isfile(os.path.join(self.root, 'splits.json'))


