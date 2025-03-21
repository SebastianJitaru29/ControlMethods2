import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Union,List, Tuple


def get_dataloaders(datadirs, batch_size, steps=1 ,train_split=0.8):
    dataset = PandaMotionDataset(datadirs, steps)
    
    train_len = (int)(len(dataset) * train_split)
    indices = np.arange(0, len(dataset), dtype=int)

    train_index = np.random.choice(indices, train_len, replace=False)
    val_index = [idx for idx in indices if idx not in train_index]

    print(len(dataset), len(train_index), len(val_index))
    train_set = PandaMotionDataset(datadirs, steps, train_index)
    val_set = PandaMotionDataset(datadirs, steps, val_index)

    return DataLoader(train_set, batch_size, True), \
           DataLoader(val_set, batch_size, True)


class PandaMotionDataset(Dataset):

    def __init__(
            self,
            datadirs: List[str],
            steps: Union[int, Tuple[int, int]] = 1,
            index_mapping = None,
        ):

        self.datadirs = datadirs
        """
        Every trajectory has 3 files: effort, positions and velocities
        All three are needed for each joint in order to calculate the final
        lagrangian.
        The rows in one trajectory corresp6/datasets/panda_trajectoriesond to the same observations.

        TODO:
        - normalization
        """
        self.index_mapping = index_mapping

        if isinstance(steps, int):
            self.step = steps
            self.steps_max = steps
            self.range = None
        else:
            self.steps_max = steps[1]
            self.steps_min = steps[0]
            self.range = range(self.steps_min, self.steps_max+1)

        self.trajectories = [self._extract_trajectory(traj) for traj in datadirs]
        self.len = sum(dct['len'] for dct in self.trajectories) if self.index_mapping is None \
                                                                else len(index_mapping)

    def __len__(self):
        return self.len
    

    def __getitem__(self, index):

        if self.index_mapping is not None:
            index = self.index_mapping[index]

        traj_idx, idx = self._map_index(index)
        if self.range is None:
            idx_next = idx + self.step
        else:
            idx_next = idx + np.random.choice(self.range)

        dfs = self.trajectories[traj_idx]

        d_time = self._get_deltatime(dfs['pos'], idx, idx_next)  # All df have same time

        q = self._get_joints(dfs['pos'], idx)
        q_dot = self._get_joints(dfs['vel'], idx)

        q = q / torch.pi
        q_dot = q_dot / (2 * torch.pi)
        data_src = torch.stack((q, q_dot))

        q_next = self._get_joints(dfs['pos'], idx_next)
        q_dot_next = self._get_joints(dfs['vel'], idx_next)


        q_next = q_next / torch.pi
        q_dot_next = q_dot_next / (2 * torch.pi)
        data_target = torch.stack((q_next, q_dot_next))

        efforts = self._get_joints(dfs['eff'], idx)


        # TODO: get deltatime, format the information q, q_dot (2, 7)
        # TODO: format with dt and torque for easy training

        return (data_src, efforts, torch.tensor(d_time).unsqueeze(0)), \
                data_target


    def _map_index(self, index):
        traj_idx = 0
        for traj_idx, traj in enumerate(self.trajectories):
            if index >= traj['len']:
                index -= traj['len']
            else:
                break
        return traj_idx, index

    def _extract_trajectory(self, traj_dir):

        dfs = {}
        dfs['pos'] = pd.read_csv(f'{traj_dir}/positions')
        dfs['vel'] = pd.read_csv(f'{traj_dir}/velocities')
        dfs['eff'] = pd.read_csv(f'{traj_dir}/efforts')
        dfs['len'] = len(dfs['pos']) - self.steps_max

        return dfs

    @staticmethod
    def _get_deltatime(df, idx_first, idx_last):

        t_first = df.iloc[idx_first]['sec'] + df.iloc[idx_first]['nsec'] * 1e-9
        t_last = df.iloc[idx_last]['sec'] + df.iloc[idx_last]['nsec'] * 1e-9

        return t_last - t_first

    @staticmethod
    def _get_joints(df, idx):
        values = df.iloc[idx].values

        # Drop sec and nsec and format as tensor
        return torch.tensor(values[:-2], dtype=torch.float64)