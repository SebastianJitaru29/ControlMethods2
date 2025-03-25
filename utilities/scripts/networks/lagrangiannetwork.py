import torch
import torch.nn as nn

from .trilnetwork import TrilNetwork
from .neuralnetwork import NeuralNetwork


class LagrangianNetwork(nn.Module):

    # TODO define hyperparameters
    def __init__(
            self,
            *,
            lagrangian: callable=None,
            integrator: callable=None,
            device: str='cpu'
        ):           
        super(LagrangianNetwork, self).__init__()
        print(device)

        self.network_mass = TrilNetwork(7, 7, [128, 128], device=device)
        self.network_dissipation = TrilNetwork(7, 7, [128, 128], device=device)

        # TODO should this be 
        self.network_ep = NeuralNetwork(7, 7, [128, 128], device=device)
        # network_action = NeuralNetwork(2, [7, 7], [256, 256])

        # TODO not an actual part of the proposed architecture
        self.network_coriolis = NeuralNetwork(14, [7, 7], [128, 128], device=device)

        self.lagrangian = lagrangian if lagrangian is not None \
                                     else self.arm_lagrangian

        self.integrator = integrator if integrator is not None \
                                     else self.base_integrator


    # TODO dicts to select which things to use?
    # torques, dt
    def forward(self, x, torques, d_time):
        """"""
        q = x[:, 0]
        q_dot = x[:, 1]

        mass_q = self.network_mass(q)
        dissipation_q = self.network_dissipation(q)
        energy_p = self.network_ep(q)

        frmt = torch.concat((q, q_dot), dim=1)
        coriolis = self.network_coriolis(frmt)

        # kinetic energy = 1/2 q_dot_T * mass_q * q_dot^2 or smth
        # potential energy is predicted
        # L = T - V -> d/dt (dL/dq_dot) - dL/dq = F_ext
        # F_ext = Dq + u

        # q_dotdot = mass_q_inv (A*u - C*q_dot - G(q) - D*q_dot)
        
        # How about adding coriolis prediction matrix, depends on both position and speed
        # What are the properties of the coriolis matrix? -> tril or normal network?

        ## where do the C and G come from?

        q = q.double()
        q_dot = q_dot.double()

        q_dotdot_hat = self.lagrangian(
            mass_q=mass_q,
            coriolis=coriolis,
            energy_p=energy_p,
            dissipation_q=dissipation_q,
            q_dot=q_dot,
            torques=torques,
        ).squeeze(2)
        

        q_dot_hat = self.integrator(q_dot, q_dotdot_hat, d_time)
        q_hat = self.integrator(q, (q_dot_hat + q_dot) / 2, d_time)
        
        return torch.stack((q_hat, q_dot_hat), dim=1)
    

    # TODO acceleration at next time step, integrate
    #      towards that. How to know acc now?
    @staticmethod
    def base_integrator(x, x_dot, d_time):
        """Returns q_dot_hat t+1 based on..."""
        x_hat = x + x_dot * d_time
        # TODO add more proper integrator
        
        return x_hat
    
    @staticmethod
    def arm_lagrangian(mass_q, coriolis, energy_p,
                       dissipation_q, q_dot, torques):
        """"""
        # TODO q_dot might need unsqueezing
        q_dot = q_dot.unsqueeze(2)

        cor_dot = coriolis @ q_dot
        diss_dot = dissipation_q @ q_dot

        precomp = (torques
                  - cor_dot.squeeze(2)
                  - energy_p
                  - diss_dot.squeeze(2))
        
        #print(precomp.shape)
        
        return (mass_q.inverse() @ precomp.unsqueeze(2))



        