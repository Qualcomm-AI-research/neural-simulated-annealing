# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


import os
import pickle
import time

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

from neuralsa.configs import NeuralSAExperiment
from neuralsa.model import BinPackingActor, KnapsackActor, TSPActor
from neuralsa.problem import TSP, BinPacking, Knapsack
from neuralsa.sa import sa

# For reproducibility on GPU
torch.backends.cudnn.deterministic = True


def create_folder(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created: {dirname}")


def save(obj, path):
    with open(path + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=NeuralSAExperiment, group="experiment")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: NeuralSAExperiment) -> None:
    if "cuda" in cfg.device and not torch.cuda.is_available():
        cfg.device = "cpu"
        print("CUDA device not found. Running on cpu.")

    # Set Problem and Networks
    if cfg.problem == "knapsack":
        if cfg.capacity is None:
            # Set capacity as done in the paper
            if cfg.problem_dim == 50:
                cfg.capacity = 12.5
            elif cfg.problem_dim == 100:
                cfg.capacity = 25
            else:
                cfg.capacity = cfg.problem_dim / 8
        problem = Knapsack(cfg.problem_dim, device=cfg.device, params={"capacity": cfg.capacity})
        actor = KnapsackActor(cfg.embed_dim, device=cfg.device)
    elif cfg.problem == "binpacking":
        problem = BinPacking(cfg.problem_dim, device=cfg.device)
        actor = BinPackingActor(cfg.embed_dim, device=cfg.device)
    elif cfg.problem == "tsp":
        problem = TSP(cfg.problem_dim, device=cfg.device)
        actor = TSPActor(cfg.embed_dim, device=cfg.device)
    else:
        raise ValueError("Invalid problem name.")

    if cfg.model_path is None:
        training_problem_dim = 20 if cfg.problem == "tsp" else 50
        cfg.model_path = (
            "models/" + cfg.problem + str(training_problem_dim) + "-" + cfg.training.method + ".pt"
        )

    # Load trained model
    actor.load_state_dict(torch.load(os.path.join(cfg.model_path), map_location=cfg.device))
    actor.eval()
    print("Loaded model at ", cfg.model_path)

    # Use Kool's dataset for TSP 20, 50 and 100
    if cfg.problem == "tsp" and cfg.problem_dim in [20, 50, 100]:
        filename = os.path.join(
            get_original_cwd(), cfg.data_path, "tsp" + str(cfg.problem_dim) + "_test_seed1234.pkl"
        )
        with open(filename, "rb") as f:
            tsp_test = pickle.load(f)

        cfg.n_problems = 10000  # These datasets have 10K instances
        coords = torch.tensor(tsp_test, device=cfg.device)
        problem = TSP(cfg.problem_dim, cfg.n_problems, device=cfg.device)
        problem.set_params(coords=coords)

    else:
        # Create random instances
        params = problem.generate_params(mode="test")
        params = {k: v.to(cfg.device) for k, v in params.items()}
        problem.set_params(**params)

    # Create accumulators
    # Store the minimum cost of each problem
    # Store the time taken to evaluate all instances
    train_out_greedy = {}  # Greedy Neural SA
    train_out_sampled = {}  # Sampled Neural SA
    random_out = {}  # Vanilla SA

    for m in [1, 2, 5, 10]:  # Defines the number of steps: K = m * problem_dim
        for i in [1, 2, 3, 4, 5]:  # Different runs
            # Set seeds
            torch.manual_seed(i)
            actor.manual_seed(i)

            cfg.sa.outer_steps = cfg.problem_dim * m
            # If TSP K = m * problem_dim^2
            if cfg.problem == "tsp":
                cfg.sa.outer_steps = cfg.sa.outer_steps * cfg.problem_dim

            # Define temperature decay parameter as a function of the number of steps
            alpha = np.log(cfg.sa.stop_temp) - np.log(cfg.sa.init_temp)
            cfg.sa.alpha = np.exp(alpha / cfg.sa.outer_steps).item()

            # Define initial solution
            init_x = problem.generate_init_x().to(cfg.device)

            if m == 10:
                # Evaluate vanilla SA
                torch.cuda.empty_cache()
                start_time = time.time()
                out = sa(actor, problem, init_x, cfg, replay=None, baseline=True)
                random_out[m, i] = {}
                random_out[m, i]["min_cost"] = out["min_cost"]
                random_out[m, i]["time"] = time.time() - start_time

            torch.cuda.empty_cache()
            start_time = time.time()
            out = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=False)
            train_out_sampled[m, i] = {}
            train_out_sampled[m, i]["min_cost"] = out["min_cost"]
            train_out_sampled[m, i]["time"] = time.time() - start_time

            if m == 1:
                # Evaluate greedy Neural SA
                torch.cuda.empty_cache()
                start_time = time.time()
                out = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=True)
                train_out_greedy[m, i] = {}
                train_out_greedy[m, i]["min_cost"] = out["min_cost"]
                train_out_greedy[m, i]["time"] = time.time() - start_time

            res = torch.mean(train_out_sampled[m, i]["min_cost"]).item()

            print(
                str(m) + "x,",
                "K=" + str(cfg.sa.outer_steps) + ",",
                "random seed",
                i,
                "sampled:",
                "{:0.2f}".format(res),
            )

    path = os.path.join(os.getcwd(), "results", cfg.problem)
    create_folder(path)
    save(
        random_out,
        os.path.join(path, "random_out_" + str(cfg.problem_dim) + "-" + cfg.training.method),
    )
    save(
        train_out_sampled,
        os.path.join(path, "train_out_sampled_" + str(cfg.problem_dim) + "-" + cfg.training.method),
    )
    save(
        train_out_greedy,
        os.path.join(path, "train_out_greedy_" + str(cfg.problem_dim) + "-" + cfg.training.method),
    )


if __name__ == "__main__":
    main()
