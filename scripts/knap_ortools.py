# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
import pathlib
import time
from typing import Tuple

import numpy as np
import torch
from ortools.algorithms import pywrapknapsack_solver
from tqdm import tqdm


def batch_solve_knap(
    weights: np.ndarray, values: np.ndarray, capacity: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves a batch of knapsack problems defined by 'weights', 'values' and 'capacity'.

    Parameters
    ----------
    weights, values: torch tensor of shape (n_problems, problem_dim)
        Both vectors are allowed to be floats. This function converts them to integers to call OR-tools.
    capacity: torch.tensor of shape (1,)
        We assume all problems in the batch share the same capacity.

    Returns
    -------
    cost: numpy array
        The cost (sum of values of the items in the knapsack) of each solution.
    total_weight: numpy array
        The sum of weights of the items in the knapsack for each problem.
    wall_time: numpy array
        The running time in seconds for each problem.
    """

    n_problems, problem_dim = weights.shape
    cost = np.zeros(n_problems)
    total_weight = np.zeros(n_problems)
    wall_time = np.zeros(n_problems)

    # Iterate through the problems
    for b in tqdm(range(n_problems)):
        v = values[b, :]
        w = weights[b, :]
        c = capacity
        const = int(10 ** (-torch.min(torch.floor(torch.log10(torch.cat([w, v]) + 1e-8)))))
        w = (w * const).view(1, -1).to(int).tolist()
        v = (v * const).to(int).tolist()
        c = [int(c * const)]
        cost[b], total_weight[b], wall_time[b] = solve_knap(w, v, c, const)
    return cost, total_weight, wall_time


def solve_knap(
    w: torch.Tensor, v: torch.Tensor, c: torch.Tensor, const: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    w: list of integers of size problem_dim
        The weights of each item in the problem.
    v: list of list of integers of size problem_dim (not sure why OR-Tools requires a nested list here)
        The values of each item in the problem.
    c: list of a single integer
        The capacity of knapsack in the problem.
    const: integer
        The constant used to convert the problem from float to integer.

    Returns
    -------
    cost: float
        The cost (sum of values of the items in the knapsack) of the solution.
    total_weight: float
        The sum of weights of the items in the knapsack.
    wall_time: float
        The running time in seconds.
    """
    # Create the solver
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, ""
    )

    # Run the solver
    solver.Init(v, w, c)
    start_time = time.time()
    cost = solver.Solve() / const
    wall_time = time.time() - start_time

    # Compute the sum of weights in the solution.
    total_weight = 0
    for i in range(len(v)):
        if solver.BestSolutionContains(i):
            total_weight += w[0][i]
    total_weight = total_weight / const

    return cost, total_weight, wall_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=50)
    parser.add_argument("--n_problems", type=int, default=1000)
    parser.add_argument("--capacity", type=int, default=12.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    weights = torch.rand((args.n_problems, args.dimension))
    values = torch.rand((args.n_problems, args.dimension))
    or_cost, or_total_weight, wall_time = batch_solve_knap(weights, values, capacity=args.capacity)

    pathlib.Path("results/knapsack/ortools/").mkdir(parents=True, exist_ok=True)
    np.save(
        "results/knapsack/ortools/cost_" + str(args.dimension) + "_seed" + str(args.seed) + ".npy",
        or_cost,
    )
    np.save(
        "results/knapsack/ortools/total_weight_"
        + str(args.dimension)
        + "_seed"
        + str(args.seed)
        + ".npy",
        or_total_weight,
    )
    np.save(
        "results/knapsack/ortools/wall_time_"
        + str(args.dimension)
        + "_seed"
        + str(args.seed)
        + ".npy",
        wall_time,
    )
