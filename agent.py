import copy
import datetime
import os

from typing import List

import torch
import torch.nn as nn
import nibabel as nib
from tqdm import tqdm
from datetime import datetime

from torch.utils.data import DataLoader
from utils.loss import UBCEDiceLoss
from utils.common import reshape
from utils.uc_metrics import *


class Agent(object):
    def __init__(self, model: nn.Module, device: torch.device):

        self.model = model
        self.best_model = None
        self.best_model_stage = None
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.loss = None

        self.epochs = None
        self.epoch = None
        self.updates = None
        self.update = None

    def train(
            self,
            params,
            training_datasets: List[DataLoader]):

        self.model.to(self.device)
        self.loss = UBCEDiceLoss()

        train_loader, val_loader = training_datasets

        for stage in range(1, params['STAGES'] + 1):
            self.optimizer = self.model.get_optimizer(params['LEARNING_RATE'])
            self.update = None
            self.epochs = params['EPOCHS']
            best_val_loss = 1e2
            for epoch in range(1, self.epochs + 1):
                self.epoch = epoch
                train_loss: float = self.__train_loop(train_loader)
                print('Loss: {:.4f}'.format(train_loss))
                val_loss: float = self.__val_loop(val_loader)
                print('Loss: {:.4f}'.format(val_loss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model = copy.deepcopy(self.model)

            self.__safe_checkpoint(f"./checkpoints/{stage}")
            train_loader.dataset.datasets[0].add_samples(seed=stage)
            train_loader.dataset.update_length()
            self.__update_loop(train_loader)

            self.model = copy.deepcopy(self.best_model)
            self.optimizer = self.model.get_optimizer(params['LEARNING_RATE'])
            self.epochs = 20
            best_val_loss = 1e2
            for epoch in range(1, self.epochs + 1):
                self.epoch = epoch
                train_loss: float = self.__train_loop(train_loader)
                print('Loss: {:.4f}'.format(train_loss))
                val_loss: float = self.__val_loop(val_loader)
                print('Loss: {:.4f}'.format(val_loss))
                if val_loss < best_val_loss and epoch > 10:
                    best_val_loss = val_loss
                    self.best_model = copy.deepcopy(self.model)

            self.__safe_checkpoint(f"./checkpoints/{stage}")
            self.__update_loop(train_loader)
            self.model.reset_parameters()

    def eval(self, params, test_dataset: DataLoader):

        self.__load_state_dict(params['CHECKPOINT'])
        self.model.to(self.device)
        self.__eval_loop(params['SAVE_DIR'], test_dataset)

    def __train_loop(
            self,
            train_loader: DataLoader
    ) -> float:

        self.model.train()

        loop_loss = []
        loop = tqdm(enumerate(train_loader), total=len(train_loader), )
        for i, data in loop:
            data, _ = data
            x, y, uc_map, _, _, _, _, _ = data
            y = y.unsqueeze(1)
            x, y, uc_map = x.to(self.device, dtype=torch.float), \
                           y.to(self.device, dtype=torch.float), \
                           uc_map.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()
            torch.set_grad_enabled(True)
            y_pred: torch.Tensor = self.model(x)
            loss = self.loss(y_pred, y, uc_map)
            loop_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()

            loop.set_description(f"Epoch [train] [{self.epoch}/{self.epochs}]")

        return sum(loop_loss) / len(loop_loss)

    def __val_loop(
            self,
            val_loader: DataLoader
    ) -> float:

        self.model.eval()

        loop_loss = []
        loop = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, data in loop:
            x, y, uc_map, _, _, _, _, _ = data
            y = y.unsqueeze(1)
            x, y, uc_map = x.to(self.device, dtype=torch.float), \
                   y.to(self.device, dtype=torch.float), \
                   uc_map.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()
            y_pred: torch.LongTensor = self.model(x)
            loss = self.loss(y_pred, y)
            loop_loss.append(loss.item())

            loop.set_description(f"Epoch [val] [{self.epoch}/{self.epochs}]")

        return sum(loop_loss) / len(loop_loss)

    def __update_loop(
            self,
            train_loader: DataLoader
    ):

        self.best_model.eval()
        self.best_model.enable_dropout()

        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        with torch.no_grad():
            for i, data in loop:
                data, didx = data
                x, _, _, _, _, _, idx, flag = data
                x = x.to(self.device, dtype=torch.float)
                outputs = []
                for t in range(10):
                    y_pred: torch.Tensor = self.best_model(x)
                    y_pred: torch.Tensor = torch.sigmoid(y_pred)
                    y_pred = y_pred.detach().cpu()
                    outputs.append(y_pred)
                outputs = torch.cat(outputs, dim=1)
                unc_map = prd_uncertainty(outputs.numpy())
                mean_pred = np.mean(outputs.numpy(), axis=1)

                for j in range(y_pred.shape[0]):
                    if didx[j] == 0:
                        train_loader.dataset.datasets[0].update_sample(idx[j], mean_pred[j], unc_map[j])

                loop.set_description(f"Epoch [update] [{self.update}/{self.updates}]")

    def __eval_loop(
            self,
            save_dir: str,
            test_loader: DataLoader,
            drop_out: bool = True
    ):

        self.model.eval()
        self.model.enable_dropout()

        result = []
        loop = tqdm(enumerate(test_loader), total=len(test_loader))
        with torch.no_grad():
            for i, data in loop:
                x, slice_num, shape, file_dir, _ = data
                x = x.to(self.device, dtype=torch.float)
                if drop_out:
                    outputs = []
                    for t in range(10):
                        y_pred: torch.Tensor = self.model(x)
                        y_pred: torch.Tensor = torch.sigmoid(y_pred)
                        y_pred = y_pred.detach().cpu()
                        outputs.append(y_pred)
                    outputs = torch.cat(outputs, dim=1)
                    mean_pred = np.mean(outputs.numpy(), axis=1)[0]
                    y_pred: np.ndarray = reshape(mean_pred, (shape[1], shape[0]))
                else:
                    y_pred: torch.Tensor = self.model(x)
                    y_pred: torch.Tensor = torch.sigmoid(y_pred)
                    y_pred: np.ndarray = y_pred.detach().cpu().numpy()
                    y_pred: np.ndarray = reshape(y_pred[0, 0], (shape[1], shape[0]))

                y_pred: np.ndarray = np.where(y_pred > 0.5, 1, 0)
                y_pred: np.ndarray = np.expand_dims(y_pred, axis=2)
                result.append(y_pred)
                if (slice_num[0] + 1) % shape[2] == 0:
                    result = np.concatenate(result, axis=2)
                    ni_img = nib.Nifti1Image(result, np.eye(4))
                    path = os.path.join(file_dir[0], save_dir)
                    self.__safe_nifti(ni_img, path)
                    result = []

    @staticmethod
    def __safe_nifti(data, path):
        if not os.path.exists(path):
            os.makedirs(path)

        nib.save(data, os.path.join(path, 'output.nii.gz'))

    def __safe_checkpoint(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        checkpoint = {'state_dict': self.best_model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'epoch': self.epoch + 1}
        identifier = f"{datetime.today().date()}_ep={self.epoch}_up={self.update}"
        torch.save(checkpoint, f"{path}/{identifier}.pth.tar")

    def __load_state_dict(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError('We are sorry, we could not find your checkpoint.')

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def shutdown_safe(self, path):
        self.__safe_checkpoint(path)
