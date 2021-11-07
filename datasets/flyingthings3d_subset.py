"""
Load the FlyingThings3D-subset dataset (for visualization use).
"""

import os
import os.path as osp
import numpy as np

from torch.utils.data import Dataset


class FlyingThings3DSubset(Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        args:
    """
    def __init__(self,
                 train,
                 data_root,
                 overfit_samples=None,
                 full=True,
                 interval=1):
        self.root = osp.join(data_root, 'FlyingThings3D_subset_processed_35m')
        self.train = train

        self.samples = self.make_dataset(full, interval, overfit_samples)

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded, obj_mask_loaded = self.pc_loader(self.samples[index])
        return pc1_loaded, pc2_loaded, obj_mask_loaded, self.samples[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def make_dataset(self, full, interval, overfit_samples):
        root = osp.realpath(osp.expanduser(self.root))
        root = osp.join(root, 'train') if (self.train and overfit_samples is None) else osp.join(root, 'val')
        print(root)
        all_paths = os.walk(root)
        useful_paths = sorted([item[0] for item in all_paths if len(item[1]) == 0])
        try:
            if (self.train and overfit_samples is None):
                assert (len(useful_paths) == 19640)
            else:
                assert (len(useful_paths) == 3824)
        except AssertionError:
            print('len(useful_paths) =', len(useful_paths))
            sys.exit(1)

        if overfit_samples is not None:
            res_paths = useful_paths[:overfit_samples]
        else:
            if not full:
                res_paths = useful_paths[::interval]
            else:
                res_paths = useful_paths

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path: path to a dir, e.g., home/xiuye/share/data/Driving_processed/35mm_focallength/scene_forwards/slow/0791
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(osp.join(path, 'pc1.npy'))
        pc2 = np.load(osp.join(path, 'pc2.npy'))
        obj_mask = np.load(osp.join(path, 'obj_mask.npy'))
        # multiply -1 only for subset datasets
        pc1[..., -1] *= -1
        pc2[..., -1] *= -1
        pc1[..., 0] *= -1
        pc2[..., 0] *= -1

        return pc1, pc2, obj_mask
