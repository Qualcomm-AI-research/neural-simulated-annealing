# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import random

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from neuralsa.configs import NeuralSAExperiment
from neuralsa.model import (
    BinPackingActor,
    BinPackingCritic,
    KnapsackActor,
    KnapsackCritic,
    TSPActor,
    TSPCritic,
)
from neuralsa.problem import TSP, BinPacking, Knapsack
from neuralsa.sa import sa
from neuralsa.training import EvolutionStrategies
from neuralsa.training.ppo import ppo
from neuralsa.training.replay import Replay

# For reproducibility on GPU
torch.backends.cudnn.deterministic = True


def create_folder(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created: {dirname}")


def train_es(actor, problem, init_x, es, cfg):
    with torch.no_grad():
        es.zero_updates()
        for _ in range(es.population):
            es.perturb(antithetic=True)

            # Run SA and compute the loss
            results = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=False)
            loss = torch.mean(results[cfg.training.reward])
            es.collect(loss)

        es.step(reshape_fitness=True)

    return torch.mean(torch.tensor(es.objective))


def train_ppo(actor, critic, actor_opt, critic_opt, problem, init_x, cfg):
    # Create replay to store transitions
    replay = Replay(cfg.sa.outer_steps * cfg.sa.inner_steps)
    # Run SA and collect transitions
    sa(actor, problem, init_x, cfg, replay=replay, baseline=False, greedy=False)
    # Optimize the policy with PPO
    ppo(actor, critic, replay, actor_opt, critic_opt, cfg)


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=NeuralSAExperiment, group="experiment")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: NeuralSAExperiment) -> None:
    if "cuda" in cfg.device and not torch.cuda.is_available():
        cfg.device = "cpu"
        print("CUDA device not found. Running on cpu.")

    # Define temperature decay parameter as a function of the number of steps
    alpha = np.log(cfg.sa.stop_temp) - np.log(cfg.sa.init_temp)
    cfg.sa.alpha = np.exp(alpha / cfg.sa.outer_steps).item()

    print(OmegaConf.to_yaml(cfg))

    # Set seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Set Problem and Networks
    if cfg.problem == "knapsack":
        problem = Knapsack(
            cfg.problem_dim, cfg.n_problems, device=cfg.device, params={"capacity": cfg.capacity}
        )
        actor = KnapsackActor(cfg.embed_dim, device=cfg.device)
        critic = KnapsackCritic(cfg.embed_dim, device=cfg.device)
    elif cfg.problem == "binpacking":
        problem = BinPacking(cfg.problem_dim, cfg.n_problems, device=cfg.device)
        actor = BinPackingActor(cfg.embed_dim, device=cfg.device)
        critic = BinPackingCritic(cfg.embed_dim, device=cfg.device)
    elif cfg.problem == "tsp":
        problem = TSP(cfg.problem_dim, cfg.n_problems, device=cfg.device)
        actor = TSPActor(cfg.embed_dim, device=cfg.device)
        critic = TSPCritic(cfg.embed_dim, device=cfg.device)
    else:
        raise ValueError("Invalid problem name.")

    # Set problem seed
    problem.manual_seed(cfg.seed)

    # If using PPO, initialize optimisers and replay
    if cfg.training.method == "ppo":
        actor_opt = torch.optim.Adam(
            actor.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
        critic_opt = torch.optim.Adam(
            critic.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
    elif cfg.training.method == "es":
        # Optimization specs
        optimizer = SGD(actor.parameters(), lr=cfg.training.lr, momentum=cfg.training.momentum)
        es = EvolutionStrategies(optimizer, cfg.training.stddev, cfg.training.population)
        milestones = [int(cfg.training.n_epochs * m) for m in cfg.training.milestones]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        raise ValueError("Invalid training method.")

    with tqdm(range(cfg.training.n_epochs)) as t:
        for i in t:
            # Create random instances
            params = problem.generate_params()
            params = {k: v.to(cfg.device) for k, v in params.items()}
            problem.set_params(**params)
            # Find initial solutions
            init_x = problem.generate_init_x()
            actor.manual_seed(cfg.seed)

            # Training loop
            if cfg.training.method == "ppo":
                train_ppo(actor, critic, actor_opt, critic_opt, problem, init_x, cfg)
            elif cfg.training.method == "es":
                train_es(actor, problem, init_x, es, cfg)
                scheduler.step()

            # Rerun trained model
            train_out = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=False)
            train_loss = torch.mean(train_out["min_cost"])

            t.set_description(f"Training loss: {train_loss:.4f}")

            path = os.path.join(os.getcwd(), "models")
            name = cfg.problem + str(cfg.problem_dim) + "-" + cfg.training.method + ".pt"
            create_folder(path)
            torch.save(actor.state_dict(), os.path.join(path, name))


if __name__ == "__main__":
    main()
