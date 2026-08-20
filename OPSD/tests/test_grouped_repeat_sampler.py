import unittest

from grouped_repeat_sampler import GroupedRepeatSampler


class GroupedRepeatSamplerTest(unittest.TestCase):
    def test_each_group_is_unique_and_repeated(self):
        sampler = GroupedRepeatSampler(
            17,
            unique_prompts_per_step=8,
            rollouts_per_prompt=8,
            seed=7,
        )
        values = list(sampler)
        block_size = 8 * 8

        self.assertEqual(len(values), 3 * block_size)
        for offset in range(0, len(values), block_size):
            block = values[offset : offset + block_size]
            expected = block[:8]
            self.assertEqual(len(set(expected)), 8)
            for rollout in range(8):
                self.assertEqual(block[rollout * 8 : (rollout + 1) * 8], expected)

    def test_accelerate_style_sharding_keeps_one_prompt_per_rank(self):
        world_size = 8
        rollouts = 8
        sampler = GroupedRepeatSampler(
            32,
            unique_prompts_per_step=world_size,
            rollouts_per_prompt=rollouts,
            seed=11,
        )
        global_indices = list(sampler)
        local_indices = [global_indices[rank::world_size] for rank in range(world_size)]

        for optimizer_step in range(sampler.groups_per_epoch):
            local_slice = slice(optimizer_step * rollouts, (optimizer_step + 1) * rollouts)
            prompts = []
            for rank in range(world_size):
                rank_rollouts = local_indices[rank][local_slice]
                self.assertEqual(len(rank_rollouts), rollouts)
                self.assertEqual(len(set(rank_rollouts)), 1)
                prompts.append(rank_rollouts[0])
            self.assertEqual(len(set(prompts)), world_size)

    def test_epoch_changes_order_deterministically(self):
        first = GroupedRepeatSampler(32, 8, 8, seed=5)
        second = GroupedRepeatSampler(32, 8, 8, seed=5)
        self.assertEqual(list(first), list(second))

        first.set_epoch(1)
        second.set_epoch(1)
        self.assertEqual(list(first), list(second))
        self.assertNotEqual(list(GroupedRepeatSampler(32, 8, 8, seed=5)), list(first))

    def test_rejects_too_few_unique_prompts(self):
        with self.assertRaisesRegex(ValueError, "dataset size"):
            GroupedRepeatSampler(7, 8, 8)


if __name__ == "__main__":
    unittest.main()
