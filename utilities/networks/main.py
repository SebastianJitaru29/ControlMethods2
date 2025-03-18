import os
from lagrangiannetwork import LagrangianNetwork
from pandamotiondataset import get_dataloaders
from train import ModelTrainer

import torch
from torch.nn import MSELoss
from torch.optim import AdamW

BASE_PATH = '/media/student19/73D136FD4E0A3586/datasets/panda_trajectories'
#BASE_PATH = '/media/mattias/73D136FD4E0A3586/datasets/panda_trajectories'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run():
    print('DEVICE: ' + DEVICE)
    lnn = LagrangianNetwork(device=DEVICE)
    dl_train, dl_val = get_dataloaders(
        [f'{BASE_PATH}/traj_0', f'{BASE_PATH}/traj_1', f'{BASE_PATH}/traj_2'],
        [5, 15],
        64,
    )
    trainer = ModelTrainer(
        loss_func=MSELoss(),
        acc_func=MSELoss(),
        optim_class=AdamW,
        learning_rate=7.5e-4,
        device=DEVICE,
        dl_train=dl_train,
        dl_val=dl_val,
    )

    lnn = trainer(lnn, 25)

    for x, y in dl_val:
        x = x[0][0].unsqueeze(0).to(DEVICE), \
            x[1][0].unsqueeze(0).to(DEVICE), \
            x[2][0].unsqueeze(0).to(DEVICE)
        y = y[0].unsqueeze(0).to(DEVICE)

        out = lnn(*x)

        print('------')
        print(out)
        print('------')
        print(y)
        print('------')
        print(f'Loss: {MSELoss()(out, y)}')
        break



if __name__ == '__main__':
    run()