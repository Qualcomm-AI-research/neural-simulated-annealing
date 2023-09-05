# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


import argparse
import pathlib
from typing import Dict, Tuple

import numpy as np
import torch
from ortools.linear_solver import pywraplp
from tqdm import tqdm


def create_or_data(weights: torch.Tensor, const: float) -> Dict:
    """
    Convert a torch tensor with weights into a dictionary with the necessary
    data to run the OR-Tools solver.

    Parameters
    ----------
    weights: torch tensor of shape (problem_dim,)
    const: The constant used to convert the problem from float to integer.

    Returns
    -------
    data: dict
        A dictionary containing the parameters of the problem.
    """
    data = {}
    weights = (weights * const).int()
    data["weights"] = weights.tolist()
    data["items"] = list(range(len(weights)))
    data["bins"] = data["items"]
    data["bin_capacity"] = const
    return data


def batch_solve_bin(
    weights: np.ndarray, budget: int = 60000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a SCIP solver on the bin packing problems defined by weights.
    The budget refers to the amount of time allowed for computation in ms.

    Parameters
    ----------
    weights: torch tensor of shape (n_problems, problem_dim)
        The weights are allowed to be floats. This function converts them to integers to call OR-tools.
    budget: int
        The maximum time (in ms) the solver has to solve a problem.

    Returns
    -------
    cost: numpy array
        The cost (sum of values of the items in the knapsack) of each solution.
    optimal: numpy array
        Whether the solver has found an optimal solution.
        -1: No feasible solution found.
        0: Feasible solution found.
        1: Optimal solution found.
    best_bound: numpy array
        The best (lower) bound the solver has found up to the end of the run.
    wall_time: numpy array
        The running time in seconds for each problem.
    """
    n_problems, dimension = weights.shape  # Number of different problems inside weights
    # Accumulators
    optimal = np.zeros(n_problems)
    cost = np.zeros(n_problems)
    best_bound = np.zeros(n_problems)
    wall_time = np.zeros(n_problems)
    for b in tqdm(range(n_problems)):
        # Compute the minimum value we need to multiply the weights by to get integers
        const = int(10 ** (-torch.min(torch.floor(torch.log10(weights[b, :])))))
        data = create_or_data(weights[b, :], const)
        cost[b], optimal[b], best_bound[b], wall_time[b] = solve_bin(data, budget)
    return cost, optimal, best_bound, wall_time


def solve_bin(data: Dict, budget: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves a single instance of Bin Packing Problem.

    data: dict
        The data associated to the problem. See create_or_data above.
    budget: integer
        The time limit for the solver in ms.

    Returns
    -------
    cost: float
        The cost (number of bins used) of the solution found.
    optimal: integer
        Whether the solver has found an optimal solution.
        -1: No feasible solution found.
        0: Feasible solution found.
        1: Optimal solution found.
    best_bound: float
        The best (lower) bound the solver has found up to the end of the run.
    wall_time: float
        The running time in seconds for each problem.
    """
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.set_time_limit(budget)
    # Variables
    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in data["items"]:
        for j in data["bins"]:
            x[(i, j)] = solver.IntVar(0, 1, "x_%i_%i" % (i, j))

    # y[j] = 1 if bin j is used.
    y = {}
    for j in data["bins"]:
        y[j] = solver.IntVar(0, 1, "y[%i]" % j)

    # Constraints
    # Each item must be in exactly one bin.
    for i in data["items"]:
        solver.Add(sum(x[i, j] for j in data["bins"]) == 1)

    # The amount packed in each bin cannot exceed its capacity.
    for j in data["bins"]:
        solver.Add(
            sum(x[(i, j)] * data["weights"][i] for i in data["items"])
            <= y[j] * data["bin_capacity"]
        )

    # Objective: minimize the number of bins used.
    solver.Minimize(solver.Sum([y[j] for j in data["bins"]]))
    status = solver.Solve()
    best_bound = solver.Objective().BestBound()
    wall_time = solver.wall_time()
    if status == pywraplp.Solver.FEASIBLE or status == pywraplp.Solver.OPTIMAL:
        cost = solver.Objective().Value()
    else:  # Problem not solved
        optimal = -1
        cost = len(data["bins"])
    if status == pywraplp.Solver.OPTIMAL:
        optimal = 1
    elif status == pywraplp.Solver.FEASIBLE:
        optimal = 0
    return cost, optimal, best_bound, wall_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=2000)
    parser.add_argument("--n_problems", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(0)
    weights = torch.rand((args.n_problems, args.dimension))
    or_cost, or_opt, or_best_bound, wall_time = batch_solve_bin(weights, budget=args.budget)

    pathlib.Path("results/binpacking/ortools/").mkdir(parents=True, exist_ok=True)
    np.save(
        "results/binpacking/ortools/cost_"
        + str(args.dimension)
        + "_seed"
        + str(args.seed)
        + "_budget"
        + str(int(args.budget / 1000))
        + ".npy",
        or_cost,
    )
    np.save(
        "results/binpacking/ortools/opt_"
        + str(args.dimension)
        + "_seed"
        + str(args.seed)
        + "_budget"
        + str(int(args.budget / 1000))
        + ".npy",
        or_opt,
    )
    np.save(
        "results/binpacking/ortools/best_bound_"
        + str(args.dimension)
        + "_seed"
        + str(args.seed)
        + "_budget"
        + str(int(args.budget / 1000))
        + ".npy",
        or_best_bound,
    )
    np.save(
        "results/binpacking/ortools/wall_time_"
        + str(args.dimension)
        + "_seed"
        + str(args.seed)
        + "_budget"
        + str(int(args.budget / 1000))
        + ".npy",
        wall_time,
    )
