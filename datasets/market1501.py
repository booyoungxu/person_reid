# -*- coding: utf-8 -*-

from .dataset import DataSet
import os
from utils.osutils import make_dir
import re
from glob import glob
import shutil
from .serialization import write_json


class Market1501(DataSet):
    def __init__(self, root, split_index=0, num_val=100, process=True):
        super(Market1501, self).__init__(root=root, split_index=split_index)
        if process:
            self.process()

        if not self.check_integrity():
            raise ValueError('data process error!')

        self.load(num_val)

    def process(self):
        if self.check_integrity():
            print('data have processed')
            return
        raw_path = os.path.join(self.root, 'raw')
        make_dir(raw_path)
        raw_images_dir = os.path.join(raw_path, 'Market-1501-v15.09.15')
        if not os.path.exists(raw_images_dir):
            raise ValueError("Please put Market-1501-v15.09.15 into {}".format(raw_path))

        images_dir = make_dir(os.path.join(self.root, 'images'))
        make_dir(images_dir)

        identities = [[[] for _ in range(6)] for _ in range(1502)]

        def rename(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
            fpaths = sorted(glob(os.path.join(raw_images_dir, subdir)))
            pids = set()
            for fpath in fpaths:
                fname = os.path.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                if pid == -1:   # junk images
                    continue
                assert 0 <= pid <= 1501
                assert 1 <= pid <= 6
                cam -= 1
                pids.add(pid)
                fname =  ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                shutil.copy(fpath, os.path.join(images_dir, fname))
            return pids
        trainval_pids = rename('bounding_box_train')
        query_pids =rename('query')
        gallery_pids = rename('bounding_box_test')
        assert query_pids <= gallery_pids
        assert trainval_pids.isdisjoint(gallery_pids)

        meta = {'name': 'Market1501', 'shot': 'multiple', 'cameras': 6, 'identities': identities}
        write_json(meta, os.path.join(self.root, 'meta.json'))

        splits = [{'trainval': sorted(list(trainval_pids)),
                   'query': sorted(list(query_pids)),
                   'gallery': sorted(list(gallery_pids))}]
        write_json(splits, os.path.join(self.root, 'splits.json'))










