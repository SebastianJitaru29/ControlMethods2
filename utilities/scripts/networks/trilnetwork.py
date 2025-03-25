import torch
import torch.nn as nn
from typing import List, Union
# TODO networks could have single baseclass

"""
Initial idea of networks M and D:
 - predict diagonal -> non-zeroing activation
 - predict bottom left -> no activation
 : possible to use upsampling instead?

V:
 - Single output neuron

A:
 - N*W output, reshape to matrix
"""

class TrilNetwork(nn.Module):
    """"""

    def __init__(
            self,
            input_size: int,
            matrix_dim: int,
            hidden: Union[int, List[int]],
            *_,
            leaky_alpha=0.05,
            softplus_beta=1.0,
            softplus_th=20,
            epsilon=1e-6,
            device="cpu",
        ):
        """"""
        super(TrilNetwork, self).__init__()
        
        self.eps = epsilon
        self.act = nn.LeakyReLU(leaky_alpha)
        self.act_diag = nn.Softplus(softplus_beta, softplus_th)
        self.matrix_dim = matrix_dim
        self.mask = torch.diag(torch.ones(matrix_dim, dtype=torch.bool, device=device))
        self.tril_indc = torch.tril_indices(matrix_dim, matrix_dim, device=device)

        output_size = (matrix_dim ** 2 - matrix_dim) // 2 + matrix_dim

        self.layers = []
        self.out = None

        self.device = device

        if isinstance(hidden, int):
            self._create_network_single(input_size, hidden, output_size)
        else:
            self._create_network_list(input_size, hidden, output_size)

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        """"""
        for layer in self.layers:
            x = self.act(layer(x))

        x = self.out(x)

        # TODO check for epsilon on diagonal

        # This formats the output to a lower triangular matrix
        # The diagonal activation is performend over the diagonal
        mat = torch.zeros((x.shape[0], self.matrix_dim, self.matrix_dim),
                          dtype=torch.float64, device=self.device)
        mat[:, self.tril_indc[0], self.tril_indc[1]] = x.to(torch.float64)
        mat = torch.where(self.mask, self.act_diag(mat)+self.eps, mat).double()

        return mat
    
    def _create_network_single(self, n_in: int, n_hid: int, n_out: int):
        self.layers.append(nn.Linear(n_in, n_hid, device=self.device))
        self.out = nn.Linear(n_hid, n_out, device=self.device)

    def _create_network_list(self, n_in: int, hid: List[int], n_out: int):
        self.layers.append(nn.Linear(n_in, hid[0], device=self.device))
        for idx in range(1, len(hid)):
            self.layers.append(nn.Linear(hid[idx-1], hid[idx], device=self.device))
        self.out = nn.Linear(hid[-1], n_out, device=self.device)


# Some testing with lower triangular matricis
if __name__ == '__main__':
    tens = torch.tensor([[-0.5, 10, -5, -7, 0.3, 0.1],
                         [-0.5, 10, -5, -6, 0.3, 0.1]])
    mat = torch.zeros((2, 3, 3))
    mat[:, torch.tril_indices(3, 3)[0], torch.tril_indices(3, 3)[1]] = tens

    mask = torch.diag(torch.ones(3, dtype=torch.bool))
    print(mask)
    act_diag = nn.Softplus()
    mat = torch.where(mask, act_diag(mat), mat)
    print(mat)

    network_deep = TrilNetwork(2, 3, [128, 128])
    network_shallow = TrilNetwork(2, 3, 128)
    rand_in = torch.randn(2, 2)
    print('------------')
    print(rand_in)
    print('------------')
    print(network_deep(rand_in))
    print('------------')
    print(network_shallow(rand_in))

