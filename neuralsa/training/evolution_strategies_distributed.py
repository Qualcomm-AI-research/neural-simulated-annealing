# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


import os

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Optimizer

from neuralsa.training import EvolutionStrategies


def run_es(world_size, forward, args=()):
    # Create environment
    mp.set_start_method("spawn")
    queue = mp.Queue()

    args = (world_size, queue) + args

    # Begin
    print(f"Spawning {world_size} environments")
    mp.spawn(forward, args=args, nprocs=world_size, join=True)

    return queue.get()


def setup(rank, world_size):
    """Setup the multiprocessing environment.

    Args:
        rank: process index
        world_size: total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """End the multiprocessing environment

    Call this at the end of your multiprocessing loop.
    """
    dist.destroy_process_group()


class EvolutionStrategiesDistributed(EvolutionStrategies):
    def __init__(self, optimizer: Optimizer, stddev: float, population: int, rank: int) -> None:
        super().__init__(optimizer, stddev, population)

        # Initialize storage
        self.zero_updates()
        self.mean_objective = 0
        self.sync_params()
        self.storage = []
        self.rank = rank

    def __str__(self):
        return f"EvolutionStrategiesDistributed()"

    def step(self, record_loss: bool = False) -> None:
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                # Set parameter to original value
                p.data = self.parameters[idx]

                # Compute step: We index into element 0, since we do not use
                # the full list in the distributed setting
                perturbation = self.perturbations[idx][0]
                objective = self.objective[0].clone().to(perturbation)
                summand = perturbation * objective

                # Collect all the partial summands
                dist.all_reduce(summand)
                dist.all_reduce(objective)

                # HACK: Manually write in a gradient
                p.grad = summand / (self.stddev * self.population)

                idx += 1

                # Convert objective to a mean for storing
                self.mean_objective = objective / self.population

        self.optimizer.step()

        if record_loss and (self.rank == 0):
            self.storage.append(self.mean_objective)

    def sync_params(self, src: int = 0) -> None:
        """Synchronize model parameters from model rank=src.

        Args:
            src: source process for model parameters
        """
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                dist.broadcast(p, src)

    def cleanup(self, queue: mp.Queue):
        if self.rank == 0:
            queue.put(np.array(self.storage))
