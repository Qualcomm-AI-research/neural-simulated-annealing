# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch

from neuralsa.utils import repeat_to


class Problem(ABC):
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.generator = torch.Generator(device=device)

    def gain(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.cost(s) - self.cost(self.update(s, a))

    def manual_seed(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    @abstractmethod
    def cost(self, s: torch.Tensor) -> torch.float:
        pass

    @abstractmethod
    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        pass

    @abstractmethod
    def generate_params(self) -> Dict[str, torch.Tensor]:
        pass

    @property
    def state_encoding(self) -> torch.Tensor:
        pass

    @abstractmethod
    def generate_init_state(self) -> torch.Tensor:
        pass

    def to_state(self, x: torch.Tensor, temp: torch.Tensor):
        return torch.cat([x, self.state_encoding, repeat_to(temp, x)], -1)

    def from_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return state[..., : self.x_dim], state[..., self.x_dim : -1], state[..., -1:]


class Rosenbrock(Problem):
    x_dim = 2

    def __init__(self, dim=2, n_problems=256, device="cpu", params={}):
        """
        Initialize random Rosenbrock functions.

        Args:
            dim: int
            a: [n_problems, 1]
            b: [n_problems, 1]
        """
        super().__init__(device)
        self.dim = dim
        self.n_problems = n_problems
        self.set_params(**params)

    def set_params(self, a=None, b=None):
        self.a = a
        self.b = b

    def generate_params(self, mode="train"):
        if mode == "test":
            self.manual_seed(0)
        a = torch.rand(self.n_problems, 1, device=self.device, generator=self.generator)
        b = 100 * torch.rand(self.n_problems, 1, device=self.device, generator=self.generator)
        return {"a": a, "b": b}

    @property
    def state_encoding(self) -> torch.Tensor:
        """Return parameters to add to state vectors"""
        return torch.cat([self.a, self.b / 100], -1)

    def generate_init_x(self) -> torch.Tensor:
        x = torch.randn(self.n_problems, self.dim, device=self.device, generator=self.generator)
        return x

    def generate_init_state(self) -> torch.Tensor:
        x = torch.randn(self.n_problems, self.dim, device=self.device, generator=self.generator)
        return torch.cat([x, self.state_encoding], -1)

    def cost(self, s: torch.Tensor) -> torch.Tensor:
        """Evaluate Rosenbrock

        Args:
            s: [n_problems, self.dim]
        Returns:
            [n_problems] costs
        """
        return torch.sum(
            self.b * (s[:, 1:] - s[:, :-1] ** 2.0) ** 2.0 + (self.a - s[:, :-1]) ** 2.0, dim=-1
        )

    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return s + a


class Knapsack(Problem):
    x_dim = 1

    def __init__(
        self,
        dim: int = 50,
        n_problems: int = 256,
        device: str = "cpu",
        params: Dict[str, torch.Tensor] = {},
    ) -> None:
        """Weights, values and states: 2D [batch size, num_items]"""
        super().__init__(device)
        self.dim = dim
        self.n_problems = n_problems
        params["capacity"] = params["capacity"] * torch.ones((self.n_problems, 1))
        self.set_params(**params)

    def set_params(
        self,
        weights: torch.Tensor = None,
        values: torch.Tensor = None,
        capacity: torch.Tensor = None,
    ):
        self.weights = weights
        self.values = values
        self.capacity = capacity

    def generate_params(self, mode: str = "train") -> Dict[str, torch.Tensor]:
        if mode == "test":
            self.manual_seed(0)
        v = torch.rand(self.n_problems, self.dim, device=self.device, generator=self.generator)
        w = torch.rand(self.n_problems, self.dim, device=self.device, generator=self.generator)
        if self.capacity is not None:
            c = self.capacity
        else:
            c = (
                self.dim
                * (
                    1
                    + torch.rand((self.n_problems, 1), device=self.device, generator=self.generator)
                )
                / 8
            )
        return {"values": v, "weights": w, "capacity": c}

    @property
    def state_encoding(self) -> torch.Tensor:
        """
        Returns:
            [batch_size, dim, 2] tensor
        """
        ones = torch.ones((self.dim,), device=self.device)
        capacity = self.capacity * ones / self.dim
        return torch.stack([self.weights, self.values, capacity], -1)

    def generate_init_x(self) -> torch.Tensor:
        x = torch.zeros((self.n_problems, self.dim, 1), device=self.device)
        return x

    def generate_init_state(self) -> torch.Tensor:
        x = self.generate_init_x()
        return torch.cat([x, self.state_encoding], -1)

    def cost(self, s: torch.Tensor) -> torch.Tensor:
        v = torch.sum(self.values * s[..., 0], -1)
        w = torch.sum(self.weights * s[..., 0], -1)
        return -v * (w < self.capacity[..., 0])

    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Computes XOR function (N.B. bitwise ops don't work on floats)"""
        return ((s > 0.5) ^ (a > 0.5)).float()


class BinPacking(Problem):
    x_dim = 1

    def __init__(
        self,
        dim: int = 50,
        n_problems: int = 256,
        device: str = "cpu",
        params: Dict[str, torch.Tensor] = {},
    ):
        """Initialize BinPacking.

        Args:
            dim: num items
            n_problems: batch size
            params: {'weight': torch.Tensor}
        """
        super().__init__(device)
        self.dim = dim
        self.n_problems = n_problems
        self.set_params(**params)

    def set_params(self, weights: torch.Tensor = None):
        """Set params.

        Args:
            weights: [batch size, dim]
        """
        self.weights = weights
        self.capacity = 1

    def generate_params(self, mode: str = "train") -> Dict[str, torch.Tensor]:
        """Generate random weights in [0,1). Capacity taken to be 1.

        Returns:
            weights [num problems]
        """
        # WLOG capacity taken to be 1
        if mode == "test":
            torch.manual_seed(0)
        w = torch.rand(self.n_problems, self.dim, device=self.device, generator=self.generator)
        return {"weights": w}

    def cost(self, s: torch.Tensor):
        """Compute bin-packing objective (lower is better)

        K = num occupied bins

        Args:
            s: [batch size, dim (item)]
        """
        x = s[..., 0].long()

        # Num occupied bins
        volumes = self.get_bin_volume(x)
        occupied = (volumes > 0).float()
        overflowed = (volumes > 1).float()

        K = torch.sum(occupied, -1)
        overflowed = (torch.sum(overflowed, -1) > 0.5).float()

        # If overflowed set to max vol
        return overflowed * self.dim + (1 - overflowed) * K

    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Take bin assignment vector and update

        e.g. [1,5,2,3] x [0,2,0,0] -> [1,2,2,3]
        """
        idx, src = a[:, 0], a[:, 1]
        x = torch.scatter(s[..., 0], -1, idx[:, None], src[:, None].float())

        return x[..., None]

    @property
    def state_encoding(self) -> torch.Tensor:
        return self.weights[..., None]

    def to_state(self, x: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        batch_size, problem_dim, _ = x.shape
        # Get weight of each item
        w = self.weights.unsqueeze(2)
        # Get the free capacity of each bin
        wb = self.get_bin_volume(x[..., 0])
        free_capacity = (1 - wb).unsqueeze(2)
        # Concatenate all features
        temp = repeat_to(temp, free_capacity)
        state = torch.cat((x, w, free_capacity, temp), dim=-1)
        return state

    def generate_init_x(self) -> torch.Tensor:
        """State encoding has dims

        [state enc] = [batch size, num items, concat]
        """
        x = torch.arange(self.dim, device=self.device) * torch.ones(
            (self.n_problems, 1), device=self.device
        )
        return x[..., None]

    def generate_init_state(self) -> torch.Tensor:
        """State encoding has dims

        [state enc] = [batch size, num items, concat]
        """
        x = self.generate_init_x()
        return torch.cat([x, self.state_encoding], -1)

    def get_bin_volume(self, x: torch.Tensor) -> torch.Tensor:
        """Volume in each bin

        Args:
            x: [..., items] each element indexes into a bin
        Returns:
            [..., 1, bins] tensor
        """
        volumes = torch.zeros(x.shape, device=self.device).float()
        volumes.scatter_add_(-1, x.long(), self.weights)
        return volumes

    def get_item_weight(self, item: torch.Tensor) -> torch.Tensor:
        """Weight of item

        Args:
            item: [...]
        Returns:
            [...] tensor
        """
        # return torch.sum(item * self.weights, -1)
        return self.weights[torch.arange(len(item)), item]

    def get_item_bin_volume(self, x: torch.Tensor) -> torch.Tensor:
        """Volume of the bin the current item is in

        Args:
            X: [..., items, bins]
        Returns:
            [..., items, 1] tensor
        """
        volumes = self.get_bin_volume(x)
        return torch.gather(volumes, -1, x.long())


class TSP(Problem):
    x_dim = 1

    def __init__(self, dim: int = 50, n_problems: int = 256, device: str = "cpu", params: str = {}):
        """Initialize BinPacking.

        Args:
            dim: num items
            n_problems: batch size
            params: {'weight': torch.Tensor}
        """
        super().__init__(device)
        self.dim = dim
        self.n_problems = n_problems
        self.set_params(**params)

    def set_params(self, coords: torch.Tensor = None) -> None:
        """Set params.

        Args:
            coords: [batch size, dim, 2]
        """
        self.coords = coords

    def generate_params(self, mode: str = "train") -> Dict[str, torch.Tensor]:
        """Generate random coordinates in the unit square.

        Returns:
            coords [batch size, num problems, 2]
        """
        if mode == "test":
            self.manual_seed(0)
        coords = torch.rand(
            self.n_problems, self.dim, 2, device=self.device, generator=self.generator
        )
        return {"coords": coords}

    def cost(self, s: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean tour lengths from city permutations

        Args:
            s: [batch size, dim]
        """
        # Edge lengths
        edge_lengths = self.get_edge_lengths_in_tour(s)
        return torch.sum(edge_lengths, -1)

    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Silly city swap for now

        Args:
            s: perm vector [batch size, coords]
            a: cities to swap ([batch size], [batch size])
        """
        return self.two_opt(s[..., 0], a)[..., None]

    def two_opt(self, x: torch.Tensor, a: torch.Tensor):
        """Swap cities a[0] <-> a[1].

        Args:
            s: perm vector [batch size, coords]
            a: cities to swap ([batch size], [batch size])
        """
        # Two-opt moves invert a section of a tour. If we cut a tour into
        # segments a and b then we can choose to invert either a or b. Due
        # to the linear representation of a tour, we choose always to invert
        # the segment that is stored contiguously.
        l = torch.minimum(a[:, 0], a[:, 1])
        r = torch.maximum(a[:, 0], a[:, 1])
        ones = torch.ones((self.n_problems, 1), dtype=torch.long, device=self.device)
        fidx = torch.arange(self.dim, device=self.device) * ones
        # Reversed indices
        offset = l + r - 1
        ridx = torch.arange(0, -self.dim, -1, device=self.device) + offset[:, None]
        # Set flipped section to all True
        flip = torch.ge(fidx, l[:, None]) * torch.lt(fidx, r[:, None])
        # Set indices to replace flipped section with
        idx = (~flip) * fidx + flip * ridx
        # Perform 2-opt move
        return torch.gather(x, 1, idx)

    @property
    def state_encoding(self) -> torch.Tensor:
        return self.coords

    def get_coords(self, s: torch.Tensor) -> torch.Tensor:
        """Get coords from tour permutation."""
        permutation = s[..., None].expand_as(self.coords).long()
        return self.coords.gather(1, permutation)

    def generate_init_x(self) -> torch.Tensor:
        perm = torch.cat(
            [
                torch.randperm(self.dim, device=self.device, generator=self.generator).view(1, -1)
                for _ in range(self.n_problems)
            ],
            dim=0,
        ).to(self.device)
        return perm[..., None]

    def generate_init_state(self) -> torch.Tensor:
        """State encoding has dims

        [state enc] = [batch size, num items, concat]
        """
        perm = self.generate_init_x()
        return torch.cat([perm, self.state_encoding], -1)

    def get_edge_offsets_in_tour(self, s: torch.Tensor) -> torch.Tensor:
        """Compute vector to right city in tour

        Args:
            s: [batch size, dim]
        Returns:
            [batch size, dim, 2]
        """
        # Gather dataset in order of tour
        d = self.get_coords(s[..., 0])
        d_roll = torch.roll(d, -1, 1)
        # Edge lengths
        return d_roll - d

    def get_edge_lengths_in_tour(self, s: torch.Tensor) -> torch.Tensor:
        """Compute distance to right city in tour

        Args:
            s: [batch size, dim, 1]
        Returns:
            [batch size, dim]
        """
        # Edge offsets
        offset = self.get_edge_offsets_in_tour(s)
        # Edge lengths
        return torch.sqrt(torch.sum(offset**2, -1))

    def get_neighbors_in_tour(self, s: torch.Tensor) -> torch.Tensor:
        """Return distances to neighbors in tour.

        Args:
            s: [batch size, dim, 1] vector
        """
        right_distance = self.get_edge_lengths_in_tour(s)
        left_distance = torch.roll(right_distance, 1, 1)
        return torch.stack([right_distance, left_distance], -1)
