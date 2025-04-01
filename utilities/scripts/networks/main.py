import os
from lagrangiannetwork import LagrangianNetwork
from pandamotiondataset import get_dataloaders
from train import ModelTrainer

import torch
from torch.nn import MSELoss
from torch.optim import AdamW

import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = '/home/student19/catkin_ws/src/data'
# BASE_PATH = '/media/mattias/73D136FD4E0A3586/datasets/panda_trajectories'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_hold = MSELoss(reduction='sum')

def loss_func_split(x, y):
    loss_one = loss_hold(x[:, 0], y[:, 0])
    loss_two = loss_hold(x[:, 1], y[:, 1])
    return (loss_one + loss_two) / x.shape[0]

def run():
    print('DEVICE: ' + DEVICE)
    lnn = LagrangianNetwork(device=DEVICE)
    dl_train, dl_val = get_dataloaders(
        datadirs=[f'{BASE_PATH}/traj_0', f'{BASE_PATH}/traj_1', f'{BASE_PATH}/traj_2'],
        batch_size=64,
        steps=1,
    )

    trainer = ModelTrainer(
        loss_func=loss_func_split,
        acc_func=MSELoss(),
        optim_class=AdamW,
        learning_rate=1e-3,
        device=DEVICE,
        dl_train=dl_train,
        dl_val=dl_val,
        model_dir='/home/student19/catkin_ws/src/utilities/scripts/networks/models'
    )

    lnn = trainer(lnn, 100)

    output = np.zeros((96, 2, 7))
    targets = np.zeros((96, 2, 7))

    lnn.eval()
    with torch.no_grad():
        for idx, data in enumerate(dl_val):

            if idx == 96:
                break
            x, y = data
            x = x[0][0].unsqueeze(0).to(DEVICE), \
                x[1][0].unsqueeze(0).to(DEVICE), \
                x[2][0].unsqueeze(0).to(DEVICE)
            y = y[0].unsqueeze(0).to(DEVICE)

            out = lnn(*x)

            output[idx] = out.detach().cpu().numpy()
            targets[idx] = y.detach().cpu().numpy()

            """
            print('------')
            print(out)
            print('------')
            print(y)
            print('------')
            print(f'Loss: {MSELoss()(out, y)}')"
            """
        
    fig, ax = plt.subplots(2, 7, figsize=(16, 4))

    for row in range(2):
        name = 'position' if row == 0 else 'velocity'
        lims = [-1, 1] if row == 0 else [-5, 5]
        # lims = [-np.pi, np.pi] if row == 0 else [-2*np.pi, 2*np.pi]
        for col in range(7):
            ax[row][col].scatter(targets[:, row, col], output[:, row, col])
            ax[row][col].set_xlabel(f'target {name} (rad)')
            ax[row][col].set_ylabel(f'predicted {name} (rad)')
            ax[row][col].set_xlim(lims)
            ax[row][col].set_ylim(lims)
            ax[row][col].plot(lims, lims, color='red')
            
    plt.show()


if __name__ == '__main__':
    run()