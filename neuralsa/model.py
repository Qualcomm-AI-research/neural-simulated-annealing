# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralsa.utils import extend_to, repeat_to


class SAModel(nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.generator = torch.Generator(device=device)

    def manual_seed(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def get_logits(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, state: torch.Tensor, greedy: bool = False, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def baseline_sample(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


class KnapsackActor(SAModel):
    def __init__(self, embed_dim: int, device: str) -> None:
        super().__init__(device)
        self.state_dim = 5

        # Mean and std computation
        self.embed = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=True, device=device),
        )

        self.embed.apply(self.init_weights)

    def get_logits(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Neural net
        return self.embed(state)[..., 0]

    def sample_from_logits(
        self, logits: torch.Tensor, mask: torch.Tensor = None, greedy: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim = logits.shape
        if mask is not None:
            logits = logits + mask

        probs = torch.softmax(logits, dim=-1)
        if greedy:
            smpl = torch.argmax(probs, -1, keepdim=False)
        else:
            smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]

        action = F.one_hot(smpl, num_classes=problem_dim)
        return action[..., None], torch.log(probs[action.bool()])

    def baseline_sample(self, state: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert state -> x
        n_problems, problem_dim, _ = state.shape
        x = state[..., 0]
        weights = state[..., 1]
        capacity = state[..., 3] * problem_dim

        # Compute mask to avoid exceeding the knapsack's capacity by selecting too heavy items
        total_weight = torch.sum(weights * x, -1)
        free_space = capacity - extend_to(total_weight, capacity)
        oversized = (weights > free_space) * (x == 0)
        mask = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)

        logits = torch.ones((state.shape[0], problem_dim), device=state.device)
        return self.sample_from_logits(logits, mask=mask)

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert state -> x
        n_problems, problem_dim, _ = state.shape
        x = state[..., 0]
        weights = state[..., 1]
        capacity = state[..., 3] * problem_dim

        # Compute mask to avoid exceeding the knapsack's capacity
        # by selecting too heavy items
        total_weight = torch.sum(weights * x, -1)
        free_space = capacity - extend_to(total_weight, capacity)
        oversized = (weights > free_space) * (x == 0)
        mask = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)

        # Compute logits and sample an action
        logits = self.embed(state)[..., 0]
        return self.sample_from_logits(logits, mask=mask, greedy=greedy)

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, **kwargs) -> torch.Tensor:
        # Convert state -> x
        n_problems, problem_dim, _ = state.shape
        x = state[..., 0]
        weights = state[..., 1]
        capacity = state[..., 3] * problem_dim

        # Compute mask to avoid exceeding the knapsack's capacity
        # by selecting too heavy items
        total_weight = torch.sum(weights * x, -1)
        free_space = capacity - extend_to(total_weight, capacity)
        oversized = (weights > free_space) * (x == 0)
        mask = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)

        # Get logits and compute log-probabilities of the action taken
        logits = self.embed(state)[..., 0] + mask
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs[action.bool()]


class KnapsackCritic(nn.Module):
    def __init__(self, embed_dim: int, device: str = "cpu") -> None:
        super().__init__()
        self.state_dim = 5

        # Mean and std computation
        self.embed = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=True, device=device),
        )

        self.embed.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.embed(state)[..., 0].mean(-1)


class BinPackingActor(SAModel):
    def __init__(self, embed_dim: int, device: str) -> None:
        super().__init__(device)
        self.state_dim = 3

        # Mean and std computation
        self.embed_item = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=True, device=device),
        )
        # Mean and std computation
        self.embed_bin = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=True, device=device),
        )

        self.embed_item.apply(self.init_weights)
        self.embed_bin.apply(self.init_weights)

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim, _ = state.shape
        # Convert state -> x
        x = state[..., [0]]
        item_weights = state[..., [1]]
        free_capacity = state[..., [2]]
        temp = state[..., [3]]

        # Gather actions (item and bin)
        item, bin = action[:, 0], action[:, 1]

        # Get free capacity of each item's current bin
        fci = free_capacity.gather(1, x.long())

        # Build item state representation
        item_state = torch.cat([item_weights, fci, temp], -1)

        # Get item probabilities
        item_logits = self.embed_item(item_state)[..., 0]
        probs = torch.softmax(item_logits, dim=-1)
        log_probs_item = torch.log(probs.gather(1, item.view(-1, 1)))

        # Gather the weight of selected item
        item_weight = item_weights[..., 0].gather(1, item.view(-1, 1))
        item_weight = repeat_to(item_weight, free_capacity)

        # Build bin state representation
        bin_state = torch.cat([item_weight, free_capacity, temp], -1)

        # Get bin mask
        # too_big = 1 if weight > capacity
        oversized = (item_weight[..., 0] - free_capacity[..., 0]) > 0
        # Place -inf where oversized
        oversized = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        # Place -inf where item is (to avoid useless move)
        mask = torch.scatter(oversized, -1, item[..., None], torch.finfo(torch.float32).min)

        # Get bin probabilities
        bin_logits = self.embed_bin(bin_state)[..., 0]
        bin_logits = bin_logits + mask
        probs = torch.softmax(bin_logits, dim=-1)
        log_probs_bin = torch.log(probs.gather(1, bin.view(-1, 1)))

        return log_probs_item, log_probs_bin

    def sample_from_logits(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor = None,
        one_hot: bool = True,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim = logits.shape
        if mask is not None:
            logits = logits + mask

        probs = torch.softmax(logits, dim=-1)
        if greedy:
            smpl = torch.argmax(probs, -1, keepdim=False)
        else:
            smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]

        if one_hot:
            smpl = F.one_hot(smpl, num_classes=problem_dim)[..., None]
        return smpl, torch.log(probs.gather(1, smpl.view(-1, 1)))

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim, _ = state.shape
        # Convert state -> x
        x = state[..., [0]]
        item_weights = state[..., [1]]
        free_capacity = state[..., [2]]
        temp = state[..., [3]]

        # Get free capacity of each item's current bin
        fci = free_capacity.gather(1, x.long())

        # Build item state representation and get logits
        item_state = torch.cat([item_weights, fci, temp], -1)

        # Select an item: [batch, num items, 1]
        item_logits = self.embed_item(item_state)[..., 0]
        item, log_probs_item = self.sample_from_logits(item_logits, one_hot=False, greedy=greedy)

        # Gather the weight of selected item
        item_weight = item_weights[..., 0].gather(1, item.view(-1, 1))
        item_weight = repeat_to(item_weight, free_capacity)

        # Build bin state representation
        bin_state = torch.cat([item_weight, free_capacity, temp], -1)

        # Get bin mask
        # too_big = 1 if weight > capacity
        oversized = (item_weight[..., 0] - free_capacity[..., 0]) > 0
        # Place -inf where oversized
        oversized = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        # Place -inf where item is (to avoid useless move)
        mask = torch.scatter(oversized, -1, item[..., None], torch.finfo(torch.float32).min)

        # Select a bin: [batch, num bins, 1]
        bin_logits = self.embed_bin(bin_state)[..., 0]
        bin, log_probs_bin = self.sample_from_logits(
            bin_logits, mask=mask, one_hot=False, greedy=greedy
        )

        # Build action tensor
        action = torch.cat((item.view(-1, 1), bin.view(-1, 1)), dim=-1)

        return action, log_probs_item + log_probs_bin

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, **kwargs) -> torch.Tensor:
        n_problems, problem_dim, _ = state.shape
        # Convert state -> x
        x = state[..., [0]]
        item_weights = state[..., [1]]
        free_capacity = state[..., [2]]
        temp = state[..., [3]]

        # Gather actions (item and bin)
        item, bin = action[:, 0], action[:, 1]

        # Get free capacity of each item's current bin
        fci = free_capacity.gather(1, x.long())

        # Build item state representation
        item_state = torch.cat([item_weights, fci, temp], -1)

        # Get item probabilities
        item_logits = self.embed_item(item_state)[..., 0]
        probs = torch.softmax(item_logits, dim=-1)
        log_probs_item = torch.log(probs.gather(1, item.view(-1, 1)))

        # Gather the weight of selected item
        item_weight = item_weights[..., 0].gather(1, item.view(-1, 1))
        item_weight = repeat_to(item_weight, free_capacity)

        # Build bin state representation
        bin_state = torch.cat([item_weight, free_capacity, temp], -1)

        # Get bin mask
        # too_big = 1 if weight > capacity
        oversized = (item_weight[..., 0] - free_capacity[..., 0]) > 0
        # Place -inf where oversized
        oversized = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        # Place -inf where item is (to avoid useless move)
        mask = torch.scatter(oversized, -1, item[..., None], torch.finfo(torch.float32).min)

        # Get bin probabilities
        bin_logits = self.embed_bin(bin_state)[..., 0]
        bin_logits = bin_logits + mask
        probs = torch.softmax(bin_logits, dim=-1)
        log_probs_bin = torch.log(probs.gather(1, bin.view(-1, 1)))

        log_probs = log_probs_item + log_probs_bin

        return log_probs[..., 0]

    def baseline_sample(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        n_problems, problem_dim, _ = state.shape
        # Convert state -> x
        item_weights = state[..., [1]]
        free_capacity = state[..., [2]]

        # Sample item
        logits = torch.ones(state.shape[:2], device=state.device)
        item, _ = self.sample_from_logits(logits, mask=None, one_hot=False)

        # Gather the weight of selected item
        item_weight = item_weights[..., 0].gather(1, item.view(-1, 1))
        item_weight = repeat_to(item_weight, free_capacity)

        # Get bin mask
        # too_big = 1 if weight > capacity
        oversized = (item_weight[..., 0] - free_capacity[..., 0]) > 0
        # Place -inf where oversized
        oversized = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        # Place -inf where item is (to avoid useless move)
        mask = torch.scatter(oversized, -1, item[..., None], torch.finfo(torch.float32).min)

        # Sample bin
        logits = torch.ones(state.shape[:2], device=state.device)
        bin, _ = self.sample_from_logits(logits, mask=mask, one_hot=False)

        return torch.cat((item.view(-1, 1), bin.view(-1, 1)), dim=-1), None


class BinPackingCritic(nn.Module):
    def __init__(self, embed_dim: int, device: str = "cpu") -> None:
        super().__init__()
        self.q_func = nn.Sequential(
            nn.Linear(3, embed_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, 1, device=device),
        )

        self.q_func.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size, problem_dim, _ = state.shape
        q_values = self.q_func(state[..., 1:]).view(batch_size, problem_dim)
        return q_values.mean(dim=-1)


class TSPActor(SAModel):
    def __init__(self, embed_dim: int, device: str) -> None:
        super().__init__(device)
        self.c1_state_dim = 7
        self.c2_state_dim = 13

        # Mean and std computation
        self.city1_net = nn.Sequential(
            nn.Linear(self.c1_state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=False, device=device),
        )
        # Mean and std computation
        self.city2_net = nn.Sequential(
            nn.Linear(self.c2_state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=False, device=device),
        )

        self.city1_net.apply(self.init_weights)
        self.city2_net.apply(self.init_weights)

    def sample_from_logits(
        self, logits: torch.Tensor, greedy: bool = False, one_hot: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim = logits.shape
        probs = torch.softmax(logits, dim=-1)

        if greedy:
            smpl = torch.argmax(probs, -1, keepdim=False)
        else:
            smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]

        taken_probs = probs.gather(1, smpl.view(-1, 1))

        if one_hot:
            smpl = F.one_hot(smpl, num_classes=problem_dim)[..., None]

        return smpl, torch.log(taken_probs)

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim, _ = state.shape
        x, coords, temp = state[..., :1], state[..., 1:-1], state[..., [-1]]

        c1 = action[:, 0]
        # c2 = action[:, 1]

        # First city encoding
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)
        c1_state = torch.cat([coords, coords_prev, coords_next, temp], -1)

        # City 1 net
        logits = self.city1_net(c1_state)[..., 0]
        probs = torch.softmax(logits, dim=-1)
        log_probs_c1 = torch.log(probs)

        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim

        # Second city encoding: [base.prev, base, base.next,
        #                        target.prev, target, target.next]
        arange = torch.arange(n_problems)
        c1_coords = coords[arange, c1]
        c1_prev_coords = coords[arange, c1_prev]
        c1_next_coords = coords[arange, c1_next]
        base = torch.cat([c1_coords, c1_prev_coords, c1_next_coords], -1)[:, None, :]
        base = repeat_to(base, c1_state)
        c2_state = torch.cat([base, c1_state], -1)

        # City 2 net
        logits = self.city2_net(c2_state)[..., 0]
        logits[arange, c1] = -float("inf")
        logits[arange, c1_prev] = -float("inf")
        logits[arange, c1_next] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        log_probs_c2 = torch.log(probs)

        return log_probs_c1, log_probs_c2

    def baseline_sample(self, state: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None]:
        n_problems, problem_dim, _ = state.shape

        # Sample c1 at random
        logits = torch.ones((n_problems, problem_dim), device=self.device)
        c1, _ = self.sample_from_logits(logits, one_hot=False)

        # Compute mask and sample c2
        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim
        arange = torch.arange(n_problems)
        logits[arange, c1] = -float("inf")
        logits[arange, c1_prev] = -float("inf")
        logits[arange, c1_next] = -float("inf")
        c2, _ = self.sample_from_logits(logits, one_hot=False)

        # Construct action tensor and return
        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        return action, None

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim, _ = state.shape
        x, coords, temp = state[..., :1], state[..., 1:-1], state[..., [-1]]

        # First city encoding
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)
        c1_state = torch.cat([coords, coords_prev, coords_next, temp], -1)

        # City 1 net
        logits = self.city1_net(c1_state)[..., 0]
        c1, log_probs_c1 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim

        # Second city encoding: [base.prev, base, base.next,
        #                        target.prev, target, target.next]
        arange = torch.arange(n_problems)
        c1_coords = coords[arange, c1]
        c1_prev_coords = coords[arange, c1_prev]
        c1_next_coords = coords[arange, c1_next]
        base = torch.cat([c1_coords, c1_prev_coords, c1_next_coords], -1)[:, None, :]

        base = repeat_to(base, c1_state)
        c2_state = torch.cat([base, c1_state], -1)

        # City 2 net
        logits = self.city2_net(c2_state)[..., 0]
        logits[arange, c1] = -float("inf")
        logits[arange, c1_prev] = -float("inf")
        logits[arange, c1_next] = -float("inf")
        c2, log_probs_c2 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        # Construct action and log-probabilities
        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        log_probs = log_probs_c1 + log_probs_c2
        return action, log_probs[..., 0]

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, **kwargs) -> torch.Tensor:
        n_problems, problem_dim, _ = state.shape
        x, coords, temp = state[..., :1], state[..., 1:-1], state[..., [-1]]

        c1 = action[:, 0]
        c2 = action[:, 1]

        # First city encoding
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)
        c1_state = torch.cat([coords, coords_prev, coords_next, temp], -1)

        # City 1 net
        logits = self.city1_net(c1_state)[..., 0]
        probs = torch.softmax(logits, dim=-1)
        log_probs_c1 = torch.log(probs.gather(1, c1.view(-1, 1)))

        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim

        # Second city encoding: [base.prev, base, base.next,
        #                        target.prev, target, target.next]
        arange = torch.arange(n_problems)
        c1_coords = coords[arange, c1]
        c1_prev_coords = coords[arange, c1_prev]
        c1_next_coords = coords[arange, c1_next]
        base = torch.cat([c1_coords, c1_prev_coords, c1_next_coords], -1)[:, None, :]
        base = repeat_to(base, c1_state)
        c2_state = torch.cat([base, c1_state], -1)

        # City 2 net
        logits = self.city2_net(c2_state)[..., 0]
        logits[arange, c1] = -float("inf")
        logits[arange, c1_prev] = -float("inf")
        logits[arange, c1_next] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        log_probs_c2 = torch.log(probs.gather(1, c2.view(-1, 1)))

        # Construct log-probabilities and return
        log_probs = log_probs_c1 + log_probs_c2
        return log_probs[..., 0]


class TSPCritic(nn.Module):
    def __init__(self, embed_dim: int, device: str = "cpu") -> None:
        super().__init__()
        self.q_func = nn.Sequential(
            nn.Linear(7, embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, device=device),
        )

        self.q_func.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        n_problems, problem_dim, _ = state.shape
        x, coords, temp = state[..., :1], state[..., 1:-1], state[..., [-1]]
        # state encoding
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)
        state = torch.cat([coords, coords_prev, coords_next, temp], -1)
        q_values = self.q_func(state).view(n_problems, problem_dim)
        return q_values.mean(dim=-1)


class RosenNet(nn.Module):
    def __init__(self, problem_dim: int, embed_dim: int) -> None:
        super().__init__()
        state_dim = problem_dim + 1
        self.mu = torch.zeros(problem_dim)
        self.log_var = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, problem_dim),
            nn.Hardtanh(min_val=-6, max_val=2.0),
        )
        self.value_func = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def actor_evaluate(
        self, state: torch.Tensor, action: torch.Tensor, temp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, problem_dim = state.shape
        s = torch.cat((state, temp.view(-1, 1)), dim=-1)
        mu = self.mu.repeat(batch_size, 1)
        log_var = self.log_var(s)
        # Compute logprob
        log_prob = -0.5 * (
            log_var + torch.log(torch.tensor(2 * np.pi)) + torch.pow(action - mu, 2) / log_var.exp()
        )
        entropy = 0.5 * torch.log((2 * np.e * np.pi) ** problem_dim * log_var.sum(dim=-1).exp())
        return log_prob.sum(dim=1), entropy

    def critic(self, state: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        s = torch.cat((state, temp.view(-1, 1)), dim=-1)
        return self.value_func(s)

    def actor_sample(
        self, state: torch.Tensor, temp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, problem_dim = state.shape
        s = torch.cat((state, temp.view(-1, 1)), dim=-1)
        mu = self.mu.repeat(batch_size, 1)
        log_var = self.log_var(s)
        action = mu + torch.exp(0.5 * log_var) * torch.randn(
            (batch_size, problem_dim), generator=self.generator
        )
        log_prob, _ = self.actor_evaluate(state, action, temp)
        return action.view(batch_size, problem_dim), log_prob, mu, torch.exp(0.5 * log_var)
