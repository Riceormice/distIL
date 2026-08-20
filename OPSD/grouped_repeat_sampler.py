import math
import random
from collections.abc import Iterator, Sized


class GroupedRepeatSampler:
    """Arrange each optimizer step as unique prompts repeated for multiple rollouts."""

    def __init__(
        self,
        data_source: Sized | int,
        unique_prompts_per_step: int,
        rollouts_per_prompt: int,
        seed: int = 0,
    ) -> None:
        self.dataset_size = data_source if isinstance(data_source, int) else len(data_source)
        self.unique_prompts_per_step = int(unique_prompts_per_step)
        self.rollouts_per_prompt = int(rollouts_per_prompt)
        self.seed = int(seed)
        self.epoch = 0

        if self.dataset_size <= 0:
            raise ValueError("GroupedRepeatSampler requires a non-empty dataset")
        if self.unique_prompts_per_step <= 0:
            raise ValueError("unique_prompts_per_step must be positive")
        if self.rollouts_per_prompt <= 0:
            raise ValueError("rollouts_per_prompt must be positive")
        if self.dataset_size < self.unique_prompts_per_step:
            raise ValueError(
                "dataset size must be at least unique_prompts_per_step so prompts remain unique within a step"
            )

    @property
    def groups_per_epoch(self) -> int:
        return math.ceil(self.dataset_size / self.unique_prompts_per_step)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _padded_permutation(self) -> list[int]:
        indices = list(range(self.dataset_size))
        random.Random(self.seed + self.epoch).shuffle(indices)
        padded_size = self.groups_per_epoch * self.unique_prompts_per_step
        indices.extend(indices[: padded_size - len(indices)])
        return indices

    def grouped_indices(self) -> Iterator[tuple[int, ...]]:
        indices = self._padded_permutation()
        width = self.unique_prompts_per_step
        for offset in range(0, len(indices), width):
            group = tuple(indices[offset : offset + width])
            if len(set(group)) != width:
                raise RuntimeError(f"prompt group is not unique: {group}")
            yield group

    def __iter__(self) -> Iterator[int]:
        for group in self.grouped_indices():
            # Rollout-major order lets Accelerate's rank sharding assign one prompt
            # to each rank, then repeat that same assignment through accumulation.
            for _ in range(self.rollouts_per_prompt):
                yield from group

    def __len__(self) -> int:
        return self.groups_per_epoch * self.unique_prompts_per_step * self.rollouts_per_prompt
