# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


import random
from collections import deque, namedtuple
from typing import List

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "old_log_probs", "gamma")
)


class Replay:
    """
    Stores the transitions observed during training.
    """

    def __init__(self, capacity: float) -> None:
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def pop(self) -> Transition:
        return self.memory.pop()

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size > len(self.memory):
            return random.sample(self.memory, len(self.memory))
        return random.sample(self.memory, batch_size)

    def clean(self) -> None:
        self.memory = deque([], maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.memory)
