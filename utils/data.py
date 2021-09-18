import warnings
from numpy import loadtxt

from datasets.adni import ADNIData
from datasets.challenge import ChallengeData
from datasets.concat import ConcatData
from torch.utils.data import DataLoader, ConcatDataset
from utils.transforms import *


def datasets(params):
    tf = transforms() if params['AUGMENTATION'] is True else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val_uids_chl = loadtxt(params['VAL_UIDS_CHL'], dtype=str, delimiter="\n").tolist()

        train_dataset_adni = ADNIData(params, transform=tf)
        train_dataset_chl = ChallengeData(params, uids=val_uids_chl, transform=tf)
        train_dataset = ConcatData([train_dataset_adni, train_dataset_chl])
        val_dataset = ChallengeData(params, uids=val_uids_chl, phase="val")
        return train_dataset, val_dataset


def data_loader(params):
    train_dataset, val_dataset = datasets(params)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params['BATCH_TRAIN'],
        shuffle=True,
        num_workers=0,
        pin_memory=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=params['BATCH_VAL'],
        shuffle=False,
        num_workers=0,
        pin_memory=False)

    return train_loader, val_loader


def test_datasets(data, params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_uids = loadtxt(params['TEST_UIDS_ADNI'], dtype=str, delimiter="\n").tolist() \
            if data == "adni" \
            else loadtxt(params['TEST_UIDS_CHL'], dtype=str, delimiter="\n").tolist()

        test_dataset = ADNIData(params, uids=test_uids, phase='test') if data == "adni" \
            else ChallengeData(params, uids=test_uids, phase='test')

        return test_dataset


def data_loader_test(data, params):
    test_dataset = test_datasets(data, params)
    print(f"We found {str(len(test_dataset))} testing samples")

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False)

    return test_loader
