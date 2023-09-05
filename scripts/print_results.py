# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


import os
import pickle
from datetime import timedelta
from typing import Dict

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore

from neuralsa.configs import NeuralSAExperiment


def load(path: str, name: str) -> Dict:
    with open(os.path.join(path, name + ".pkl"), "rb") as f:
        return pickle.load(f)


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=NeuralSAExperiment, group="experiment")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: NeuralSAExperiment) -> None:
    # Load results
    path = os.path.join(os.getcwd(), cfg.results_path, cfg.problem)
    random_out = load(path, "random_out_" + str(cfg.problem_dim) + "-" + cfg.training.method)
    train_out_sampled = load(
        path,
        "train_out_sampled_" + str(cfg.problem_dim) + "-" + cfg.training.method,
    )
    train_out_greedy = load(
        path, "train_out_greedy_" + str(cfg.problem_dim) + "-" + cfg.training.method
    )

    # Print header
    header = "{:>8}  {:>4}  {:>24}  {:>12}".format("MODE", "K", "COST", "TIME")
    print(header)

    for m in [1, 2, 5, 10]:  # Defines the number of steps: K = m * problem_dim
        # Accumulators
        r, s, g = np.zeros(5), np.zeros(5), np.zeros(5)
        rt, st, gt = np.zeros(5), np.zeros(5), np.zeros(5)

        for i in [1, 2, 3, 4, 5]:  # Different runs
            # Accumulate results
            if m == 10:
                r[i - 1] = torch.mean(random_out[m, i]["min_cost"])
                rt[i - 1] = random_out[m, i]["time"]
            s[i - 1] = torch.mean(train_out_sampled[m, i]["min_cost"])
            st[i - 1] = train_out_sampled[m, i]["time"]
            if m == 1:
                g[i - 1] = torch.mean(train_out_greedy[m, i]["min_cost"])
                gt[i - 1] = train_out_greedy[m, i]["time"]

        # Print out mean results across the 5 runs
        # Print results for vanilla SA
        if m == 10:
            random_res = "{:>8}  {:>4}  {:>24}  {:>12}".format(
                "Random",
                str(m) + "x",
                str(np.round(np.mean(r), 3)) + " +- " + str(np.round(np.std(r), 3)),
                "{}".format(timedelta(seconds=int(np.mean(rt)))),
            )
            print(random_res)

        # Print results for Sampled Neural SA
        sampled_res = "{:>8}  {:>4}  {:>24}  {:>12}".format(
            "Sampled",
            str(m) + "x",
            str(np.round(np.mean(s), 3)) + " +- " + str(np.round(np.std(s), 3)),
            "{}".format(timedelta(seconds=int(np.mean(st)))),
        )
        print(sampled_res)

        # Print results for Greedy Neural SA
        if m == 1:
            greedy_res = "{:>8}  {:>4}  {:>24}  {:>12}".format(
                "Greedy",
                str(m) + "x",
                str(np.round(np.mean(g), 3)) + " +- " + str(np.round(np.std(g), 3)),
                "{}".format(timedelta(seconds=int(np.mean(gt)))),
            )
            print(greedy_res)


if __name__ == "__main__":
    main()
