import torch
import torch.nn as nn
import numpy as np
import torch.utils
from copy import deepcopy
import os
from typing import Union


class ModelTrainer():
    """Class to handle training loop of Neural Networks.
    
    
    """
    def __init__(
            self,
            loss_func: nn.Module,
            acc_func: nn.Module,
            optim_class: torch.nn.Module,
            learning_rate: float,
            device: torch.device,
            dl_train: torch.utils.data.DataLoader,
            dl_val: torch.utils.data.DataLoader,
            dl_test: torch.utils.data.DataLoader = None,
            early_stopping: bool = True,
            patience: int = 7,
            checkpoints: Union[bool, int] = 10,
            prss_input = None,
            prss_target = None,
            model_dir: os.PathLike = None,
    ):
        """"""

        self.loss_func = loss_func.to(device)
        self.acc_func = acc_func.to(device)
        
        self.optim_class = optim_class
        self.optimizer = None
        self.lr = learning_rate
        self.device = device

        self.dl_train = dl_train
        self.dl_val = dl_val
        self.dl_test = dl_test

        self.es = early_stopping
        self.patience = patience
        self.checkpoints = checkpoints

        self.prss_input = prss_input
        self.prss_target = prss_target

        self.model_dir = model_dir


    def __call__(self, model: nn.Module, epochs: int, save_file: str=None):
        """"""
        self.optimizer = self.optim_class(model.parameters(), self.lr)
        model = model.to(self.device).double()
        loss_min = np.inf
        unimproved = 0

        if save_file is not None:
            with open(save_file, 'w') as csv_file:
                csv_file.write('loss_train,loss_val,acc_train,acc_val\n')

        best_dict = deepcopy(model.state_dict())

        for epoch in range(epochs):

            model.train()
            loss_train, acc_train = self._epoch(model, self.dl_train)

            model.eval()
            with torch.no_grad():
                loss_val, acc_val = self._epoch(model, self.dl_val, False)

            print(f'Epoch: {epoch}')
            print(f'Train\t- Loss: {loss_train},\tAcc: {acc_train}')
            print(f'Val\t- Loss: {loss_val},\tAcc: {acc_val}')

            if save_file is not None:
                with open(save_file, 'a') as csv_file:
                    csv_file.write(f'{loss_train},{loss_val},{acc_train},{acc_val}\n')
            # TODO log values

            if loss_val < loss_min:
                # Improvement
                loss_min = loss_val
                unimproved = 0
                best_dict = deepcopy(model.state_dict())
                if self.model_dir is not None:
                    path = os.path.join(self.model_dir, 'best_model.pt')
                    torch.save(model.state_dict(), path)
            elif unimproved == self.patience:
                print(f'--Early Stopping--')
                break
            else:
                unimproved += 1

            if (
                self.model_dir is not None 
                and self.checkpoints 
                and (epoch+1) % self.checkpoints == 0
            ):
                path = os.path.join(self.model_dir, 
                                    f'checkpoint_{epoch+1}.pt')
                torch.save(model.state_dict(), path)


        model.load_state_dict(best_dict)
        if self.dl_test is not None:
            loss_test, acc_test = self._epoch(model, self.dl_test, False)
            print(f'---TESTING---')
            print(f'loss: {loss_test} - acc: {acc_test}')

        
        # TODO save best model
        return model

    def eval(self, model: nn.Module, dataset: torch.utils.data.DataLoader):
        model.to(self.device)
        return self._epoch(model, dataset, False)

    # TODO save the results when asked to
    def _epoch(self, model, loader, train: bool=True):
        """"""
        loss = []
        acc = []

        for x, y in loader:
            if self.prss_input is not None:
                x = self.prss_input(x)
            if self.prss_target is not None:
                y = self.prss_target(x)

            x = x[0].to(self.device), x[1].to(self.device), x[2].to(self.device)
            y = y.to(self.device)
            y_hat = model(*x)

            loss_batch = self.loss_func(y_hat, y)
            acc_batch = self.acc_func(y_hat, y)

            if train:
                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()

            loss.append(loss_batch.item())
            acc.append(acc_batch.item())

        loss_mean = np.mean(loss)
        acc_mean = np.mean(acc)

        return loss_mean, acc_mean
    