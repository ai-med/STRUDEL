import os

import nibabel as nib
import numpy as np

from typing import List
from glob import glob
from torch.utils.data import Dataset
from utils.common import reshape, standardize


class ChallengeData(Dataset):
    def __init__(self, params, uids=None, phase="train", transform=None):

        if phase not in ('train', 'val', 'test'):
            raise ValueError('phase should be either "train" or "test"')

        if phase == "test" and transform is not None:
            raise ValueError('you cannot apply augmentation during test phase')

        self.phase = phase
        self.samples = list()
        self.transform = transform

        slice_crop: tuple = (params['SLICE_CROP'], params['SLICE_CROP'])
        self.sample_size: tuple = (params['SAMPLE_SIZE'], params['SAMPLE_SIZE'])
        skip_dirs = ['LST', 'FSL_out']

        file_dirs = [f for f in glob(params['INPUT_CHALLENGE'] + "/*/*") if not any(dir in f for dir in skip_dirs)]
        if not file_dirs:
            raise FileNotFoundError('Empty directory')

        if phase == 'train':
            if uids is not None:
                file_dirs = [f for f in file_dirs if not any(str(uid) in f for uid in uids)]

        if phase == 'test' or phase == 'val':
            file_dirs = [f for f in file_dirs if any(str(uid) in f for uid in uids)]

        for file_dir in file_dirs:
            flair_path: str = os.path.join(file_dir, f"orig/FLAIR.nii.gz")
            t1_path: str = os.path.join(file_dir, f"pre/T1.nii.gz")
            mask_path = os.path.join(file_dir, f"wmh.nii.gz") \
                if phase == "train" or phase == "val" else None

            train_paths: List[str] = [flair_path, t1_path, mask_path]
            test_paths: List[str] = [flair_path, t1_path]
            if phase == "train" and len([path for path in train_paths if os.path.exists(path)]) != 3:
                continue
            elif phase == "test" and len([path for path in test_paths if os.path.exists(path)]) != 2:
                continue

            flair, shape = self.load_volume(flair_path)
            t1, _ = self.load_volume(t1_path)
            num_slices: int = shape[2]

            if phase == "train" or phase == "val":
                mask, _ = self.load_volume(mask_path, False)
                mask[mask == 2] = 0

                start_slice = 46 if "GE3T" in file_dir else slice_crop[0]
                self.samples += [(flair[:, :, slice_num], t1[:, :, slice_num], mask[:, :, slice_num],
                                  np.zeros_like(flair[:, :, slice_num]), slice_num, shape, file_dir, 0)
                                 for slice_num in range(start_slice, num_slices - slice_crop[1])]
            elif phase == "test":
                self.samples += [(flair[:, :, slice_num], t1[:, :, slice_num],
                                  slice_num, shape, file_dir)
                                 for slice_num in range(slice_crop[0], num_slices - slice_crop[1])]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.phase == "train" or self.phase == "val":
            flair, t1, mask, uc_map, slice_num, shape, file_dir, flag = self.samples[idx]
        else:
            flair, t1, slice_num, shape, file_dir = self.samples[idx]

        x = np.stack((flair, t1), axis=2)
        if self.transform is not None:
            x, mask = self.transform((x, mask))

        x = np.transpose(x, (2, 0, 1))
        if self.phase == "train" or self.phase == "val":
            return x, mask, uc_map, slice_num, shape, file_dir, idx, flag
        else:
            return x, slice_num, shape, file_dir, idx

    def load_volume(self, path, stdz=True):
        volume: np.ndarray = nib.load(path).get_fdata()
        shape = volume.shape
        volume: np.ndarray = volume[:, :, :, 0] if volume.ndim == 4 else volume
        volume: np.ndarray = standardize(volume) if stdz else volume
        volume: np.ndarray = reshape(volume, self.sample_size)
        return volume, shape


