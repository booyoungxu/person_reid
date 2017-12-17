# -*- coding: utf-8 -*-
import os
from PIL import Image


class ReadImages(object):
    def __init__(self, dataset, root=None, transforms=None):
        super(ReadImages, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = os.path.join(self.root, fname)
        image = Image.open(fpath).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)

        return image, fname, pid, camid