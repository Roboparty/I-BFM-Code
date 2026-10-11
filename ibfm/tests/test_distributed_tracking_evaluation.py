from __future__ import annotations

import pytest

from humanoidverse.training.workspace import (
    _contiguous_motion_shard,
    _merge_distributed_evaluation_results,
)


def test_motion_shards_cover_dataset_once_and_are_balanced() -> None:
    shards = [_contiguous_motion_shard(2719, rank, 8) for rank in range(8)]

    assert sorted(len(shard) for shard in shards) == [339, 340, 340, 340, 340, 340, 340, 340]
    assert [motion_id for shard in shards for motion_id in shard] == list(range(2719))


def test_merge_distributed_evaluation_results() -> None:
    merged = _merge_distributed_evaluation_results(
        [
            {"tracking": {"motion_0": {"motion_id": 0, "emd": 1.0}}},
            {"tracking": {"motion_1": {"motion_id": 1, "emd": 2.0}}},
        ]
    )

    assert set(merged["tracking"]) == {"motion_0", "motion_1"}
    assert merged["tracking"]["motion_1"]["motion_id"] == 1


def test_merge_rejects_duplicate_motion_metrics() -> None:
    with pytest.raises(RuntimeError, match="duplicate metric keys"):
        _merge_distributed_evaluation_results(
            [
                {"tracking": {"same_motion": {"motion_id": 0}}},
                {"tracking": {"same_motion": {"motion_id": 1}}},
            ]
        )


def test_tracking_worker_rebinds_shared_environment_motion_ids(monkeypatch) -> None:
    """A rank-local evaluation shard must replace stale training motion IDs."""
    from types import SimpleNamespace

    import torch

    from humanoidverse.agents.evaluations import humanoidverse_mjlab as evaluation

    class MotionLib:
        all_motions_loaded = True
        _motion_dt = torch.tensor([0.02, 0.02])

        @staticmethod
        def get_motion_length(ids):
            return torch.ones_like(ids, dtype=torch.float32)

    core_env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=4,
        motion_ids=torch.tensor([19, 18, 17, 16]),
        motion_len=torch.zeros(4),
        motion_start_times=torch.ones(4),
        motion_dt=torch.tensor([9.0]),
        _motion_lib=MotionLib(),
        carry_box_enabled=False,
    )
    wrapper = SimpleNamespace(_env=core_env, include_last_action=False)

    class StopAfterAssignment(RuntimeError):
        pass

    def stop_before_context(*args, **kwargs):
        raise StopAfterAssignment

    monkeypatch.setattr(evaluation, "extract_model", lambda agent: object())
    monkeypatch.setattr(evaluation, "get_backward_observation", stop_before_context)

    with pytest.raises(StopAfterAssignment):
        evaluation._async_tracking_worker(
            ([0, 1], 0, object()),
            env=wrapper,
            motions_preloaded=True,
        )

    assert core_env.motion_ids.min().item() >= 0
    assert core_env.motion_ids.max().item() < 2
    assert torch.all(core_env.motion_len == 1)
    assert torch.all(core_env.motion_start_times == 0)
    assert core_env.motion_dt is core_env._motion_lib._motion_dt
