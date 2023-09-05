# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


import torch
from torch.optim import Optimizer


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    """Find the inverse permutation array

    B = A[perm]
    A = B[inverse_permutation[perm]]

    Args:
        perm: permutation array
    Returns:
        inverse of same size
    """
    inv = torch.zeros_like(perm)
    inv[perm] = torch.arange(len(perm)).to(inv, non_blocking=True)
    return inv


class EvolutionStrategies:
    def __init__(self, optimizer: Optimizer, stddev: float, population: int) -> None:
        self.stddev = stddev
        self.optimizer = optimizer
        self.population = population
        # Initialize storage
        self.zero_updates()

    def __str__(self):
        return f"EvolutionStrategies()"

    def zero_updates(self) -> None:
        """Call to erase perturbation buffers"""
        self.optimizer.zero_grad()
        self.objective = []
        self.parameters = []
        self.perturbations = []
        self.num_perturbations = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                self.parameters.append(p.data)
                self.perturbations.append([])

    def perturb(self, antithetic: bool = False) -> None:
        """Perturb the weights of the current model.

        Args:
            antithetic: for each perturbation eps, sample a -eps
        """
        idx = 0
        self.num_perturbations += 1
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if antithetic and (self.num_perturbations % 2 == 0):
                    perturbation = -self.perturbations[idx][-1]
                else:
                    perturbation = self.stddev * torch.randn_like(p)
                self.perturbations[idx].append(perturbation)
                p.data = self.parameters[idx].data + perturbation
                idx += 1

    def collect(self, value: torch.Tensor) -> None:
        self.objective.append(value)

    def step(self, reshape_fitness: bool = False):
        """Perform an update step on the model parameters

        Args:
            reshape_fitness: fitness shaping performs a rank transformation
            on the objective values. This makes it robust against monotonic
            rescalings of the objective function
        """
        objective = torch.tensor(self.objective)

        if reshape_fitness:
            ind = torch.argsort(objective)
            rank = inverse_permutation(ind)
            n = rank.shape[-1] - 1
            objective = (2 * rank / n) - 1

        idx = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                # Set parameter to original value
                p.data = self.parameters[idx]

                # Compute step
                perturbation = torch.stack(self.perturbations[idx], -1)
                summand = objective.to(perturbation) * perturbation
                update = torch.mean(summand, -1) / self.stddev

                # HACK: Manually write in a gradient!
                p.grad = update
                idx += 1

        self.optimizer.step()

    def ghost_step(self) -> None:
        """Set params to original values"""
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                # Set parameter to original value
                p.data = self.parameters[idx]
                idx += 1
