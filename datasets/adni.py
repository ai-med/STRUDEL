import os
import random

import pandas
import nibabel as nib
import numpy as np

from typing import List
from glob import glob
from torch.utils.data import Dataset
from utils.common import reshape, standardize, set_orientation


class ADNIData(Dataset):
    def __init__(self, params, uids=None, phase="train", transform=None):

        if phase not in ('train', 'val', 'test'):
            raise ValueError('phase should be either "train", "val" or "test"')

        if phase == "test" and transform is not None:
            raise ValueError('you cannot apply augmentation during test phase')

        self.phase: str = phase
        self.samples: List = list()
        self.transform = transform

        self.slice_crop: tuple = (params['SLICE_CROP'], params['SLICE_CROP'])
        self.sample_size: tuple = (params['SAMPLE_SIZE'], params['SAMPLE_SIZE'])
        self.threshold_init: float = params['THRESHOLD_INIT']
        self.threshold_up: float = params['THRESHOLD_UP']

        self.uids = uids
        self.valid_uids: List = pandas.read_csv(params['VALID_UIDS']).FLAIR_IMAGEUID.to_list()
        self.file_dirs: List = [f for f in glob(params['INPUT_ADNI'] + "/*")
                                if any(str(uid) in f for uid in self.valid_uids)]

        if not self.file_dirs:
            raise FileNotFoundError('Empty directory')

        if self.phase == "train":
            self.train_fraction = params['FRACTION_ADNI']

        self.add_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.phase == "test":
            flair, t1, slice_num, shape, file_dir = self.samples[idx]
        else:
            flair, t1, mask, uc_map, slice_num, shape, file_dir, flag = self.samples[idx]

        x = np.stack((flair, t1), axis=2)
        if self.transform is not None:
            x, mask = self.transform((x, mask))

        x = np.transpose(x, (2, 0, 1))
        if self.phase == "test":
            return x, slice_num, shape, file_dir, idx
        else:
            return x, mask, uc_map, slice_num, shape, file_dir, idx, flag

    def load_volume(self, path, stdz=True):
        volume: np.ndarray = set_orientation(nib.load(path)).get_fdata()
        shape = volume.shape
        volume: np.ndarray = volume[:, :, :, 0] if volume.ndim == 4 else volume
        volume: np.ndarray = standardize(volume) if stdz else volume
        volume: np.ndarray = reshape(volume, self.sample_size)
        return volume, shape

    def update_sample(self, idx, y, uc_map):
        if self.phase == "test":
            raise ValueError("sorry, this function is only callable during training")

        if self.threshold_up > 0:
            y = np.where(y > self.threshold_up, 1, 0)
        flair, t1, mask, zero_map, slice_num, shape, file_dir, flag = self.samples[idx]
        if flag == 0:
            y = np.bitwise_or(y, mask)
            self.samples[idx] = (flair, t1, y, zero_map, slice_num, shape, file_dir, flag + 1)
        elif flag < 3:
            if flag == 2:
                uc_map = np.zeros_like(flair)
            self.samples[idx] = (flair, t1, y, uc_map, slice_num, shape, file_dir, flag + 1)

    def add_samples(self, uids: List = None, seed: int = None):
        if self.phase == "train":
            if self.uids is not None:
                file_dirs = [f for f in self.file_dirs if not any(str(uid) in f for uid in self.uids)]
            if self.train_fraction < 1:
                train_size = int(len(self.file_dirs) * self.train_fraction)
                print(f"We selected {str(train_size)} of {str(len(self.file_dirs))} available files")
                random.seed(seed)
                file_dirs = random.sample(self.file_dirs, k=train_size)
        else:
            if uids is None:
                print(self.uids)
                file_dirs = [f for f in self.file_dirs if any(str(uid) in f for uid in self.uids)]
            else:
                print(uids)
                file_dirs = [f for f in self.file_dirs if any(str(uid) in f for uid in uids)]

        for file_dir in file_dirs:
            head, image_uid = os.path.split(file_dir)
            head, _ = os.path.split(head)

            flair_path: str = os.path.join(file_dir, f"{str(image_uid)}_FLAIR.nii.gz")
            t1_path: str = os.path.join(file_dir, f"t1_reg.nii.gz")

            if not os.path.exists(flair_path) or not os.path.exists(t1_path):
                continue

            flair, shape = self.load_volume(flair_path)
            t1, _ = self.load_volume(t1_path)
            num_slices: int = shape[2]

            if self.phase == "test":
                self.samples += [(flair[:, :, slice_num], t1[:, :, slice_num],
                                  slice_num, shape, file_dir)
                                 for slice_num in range(self.slice_crop[0], num_slices - self.slice_crop[1])]
            else:
                mask_path = os.path.join(head, f"LST/{str(image_uid)}/ples_lpa_m{str(image_uid)}_FLAIR.nii")
                if not os.path.exists(mask_path):
                    continue

                mask, _ = self.load_volume(mask_path, False)
                mask[np.isnan(mask)] = 0
                if not np.array_equal(mask, mask.astype(bool)) and self.threshold_init > 0:
                    mask = np.where(mask > self.threshold_init, 1, 0)

                self.samples += [(flair[:, :, slice_num], t1[:, :, slice_num], mask[:, :, slice_num],
                                  np.empty_like(flair[:, :, slice_num]), slice_num, shape[0:3], file_dir, 0)
                                 for slice_num in range(self.slice_crop[0], num_slices - self.slice_crop[1])]
