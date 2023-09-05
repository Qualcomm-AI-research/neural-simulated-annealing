# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING


@dataclass
class TrainingConfig:
    method: str = "ppo"
    reward: str = "immediate"
    n_epochs: int = 1000
    lr: float = 0.0002  # learning rate
    batch_size: int = 1024
    # PPO params
    ppo_epochs: int = 10
    trace_decay: float = 0.9
    eps_clip: float = 0.25
    gamma: float = 0.9
    weight_decay: float = 0.01
    # ES params
    momentum: float = 0.9
    stddev: float = 0.05
    population: int = 16
    milestones: list = field(default_factory=lambda: [0.9])


@dataclass
class SAConfig:
    init_temp: float = 1.0
    stop_temp: float = 0.1
    outer_steps: int = 40  # number of steps at which temperature changes
    inner_steps: int = 1  # number of steps at a specific temperature
    alpha: float = MISSING  # defined as a function of init_temp and stop_temp


@dataclass
class NeuralSAExperiment:
    n_problems: int = 256  # number of problems in a batch
    problem_dim: int = 20
    embed_dim: int = 16  # size of hidden layer in the actor network

    training: TrainingConfig = field(default_factory=TrainingConfig)
    sa: SAConfig = field(default_factory=SAConfig)

    problem: str = "tsp"
    capacity: Optional[float] = field(default=None)
    device: str = "cuda:0"
    model_path: Optional[str] = field(default=None)
    results_path: str = "results"
    data_path: str = "datasets"
    seed: int = 42
