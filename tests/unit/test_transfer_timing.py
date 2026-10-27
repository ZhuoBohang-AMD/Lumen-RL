from __future__ import annotations

import ctypes
import importlib
import json
import sys
import time
import types
from types import SimpleNamespace

import pytest
import torch

from lumenrl.engine.inference.atom_teacher_ray import AtomTeacherRayActor
from lumenrl.trainer.spec_distill_trainer import SpecDistillTrainer
from lumenrl.transfer import eagle_mooncake_store
from lumenrl.transfer.eagle_mooncake_store import (
    AsyncPutManager,
    EagleMooncakeStore,
    TransferTimings,
)


def test_transfer_timings_accumulate_and_pop():
    timings = TransferTimings()
    timings.add(get_s=1.0, get_calls=1)
    timings.add(get_s=0.5, get_calls=2)

    assert timings.snapshot() == {"get_s": 1.5, "get_calls": 3.0}
    assert timings.pop() == {"get_s": 1.5, "get_calls": 3.0}
    assert timings.snapshot() == {}


class _HostBuffer:
    def __init__(self, data: bytes):
        self._raw = ctypes.create_string_buffer(data, len(data))

    def ptr(self) -> int:
        return ctypes.addressof(self._raw)


def _bare_store(mooncake) -> EagleMooncakeStore:
    store = object.__new__(EagleMooncakeStore)
    store.config = SimpleNamespace(
        get_retry_wait_seconds=0.0, get_retry_max_wait_seconds=10.0,
    )
    store.timings = TransferTimings()
    store._store = mooncake
    return store


def test_host_buffer_get_counts_misses_sleep_bytes_and_copy():
    payload = torch.arange(6, dtype=torch.int64)

    class Mooncake:
        calls = 0

        def batch_get_buffer(self, keys):
            Mooncake.calls += 1
            if Mooncake.calls == 1:
                return [None]
            return [_HostBuffer(payload.numpy().tobytes())]

    store = _bare_store(Mooncake())

    result = store._get_tensors_via_host_buffer(
        ["row_ids"], [("input_ids", (6,), torch.int64)], torch.device("cpu"),
    )

    assert torch.equal(result["input_ids"], payload)
    perf = store.timings.pop()
    assert perf["get_calls"] == 2
    assert perf["get_bytes"] == 48
    # The retry wait is clamped to at least 50 ms.
    assert perf["get_retry_sleep_s"] >= 0.04
    assert perf["get_s"] >= 0.0 and perf["copy_s"] >= 0.0


def test_remove_is_timed_even_when_it_gives_up(monkeypatch):
    class Mooncake:
        def batch_remove(self, keys, force):
            raise RuntimeError("unreachable master")

    store = _bare_store(Mooncake())
    monkeypatch.setattr(eagle_mooncake_store.time, "sleep", lambda _: None)

    store.remove_eagle3_tensors("row", has_last_hidden_states=True)

    perf = store.timings.pop()
    assert perf["remove_calls"] == 1
    assert perf["remove_s"] >= 0.0


def test_async_put_records_d2h_wait_and_rpc_time():
    class Mooncake:
        def batch_put_from(self, keys, ptrs, sizes):
            return [0] * len(keys)

    timings = TransferTimings()
    manager = AsyncPutManager(Mooncake(), timings=timings)
    try:
        manager.submit(["k_hs"], [0], [16], owner_buffer_ptr=1)
        manager.wait_for_buffer(1)
    finally:
        manager._executor.shutdown(wait=True)

    perf = timings.snapshot()
    assert set(perf) == {"put_d2h_wait_s", "put_lock_wait_s", "put_rpc_s"}
    assert all(value >= 0.0 for value in perf.values())


def _install_fake_atom(monkeypatch):
    class ModelRunner:
        def run_model(self, input_ids, batch=None):
            return "stock"

    class RLHFModelRunner(ModelRunner):
        _extract_mode = False

        def forward(self, batch):
            if self._extract_mode:
                self._store_hidden_states(batch)
            return "forward"

        def _store_hidden_states(self, batch):
            self._mooncake_store.timings.add(put_rows=1, put_wait_s=0.25)

    modules = {
        "atom": types.ModuleType("atom"),
        "atom.model_engine": types.ModuleType("atom.model_engine"),
        "atom.model_engine.model_runner": types.ModuleType("atom.model_engine.model_runner"),
        "atom.rollout": types.ModuleType("atom.rollout"),
        "atom.rollout.model_runner_ext": types.ModuleType("atom.rollout.model_runner_ext"),
    }
    modules["atom.model_engine.model_runner"].ModelRunner = ModelRunner
    modules["atom.rollout.model_runner_ext"].RLHFModelRunner = RLHFModelRunner
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(
        sys.modules, "lumenrl.engine.inference.atom_runner_ext", raising=False,
    )
    return importlib.import_module("lumenrl.engine.inference.atom_runner_ext")


def test_runner_writes_cumulative_capture_counters_from_rank_zero(monkeypatch, tmp_path):
    runner_ext = _install_fake_atom(monkeypatch)
    perf_path = tmp_path / "runner_perf.json"
    monkeypatch.setenv(runner_ext.RUNNER_PERF_FILE_ENV, str(perf_path))
    capture = SimpleNamespace(external_request_ids=["k"], num_scheduled_tokens=[3, 4])
    no_capture = SimpleNamespace(external_request_ids=[None], num_scheduled_tokens=[9])

    runner = object.__new__(runner_ext.LumenRLModelRunner)
    runner.rank = 0
    runner._extract_mode = True
    runner._mooncake_store = SimpleNamespace(timings=TransferTimings())
    runner.forward(capture)
    runner.forward(no_capture)
    runner.forward(capture)

    perf = json.loads(perf_path.read_text())
    assert perf["forwards"] == 2
    assert perf["capture_tokens"] == 14
    assert perf["put_rows"] == 3
    assert perf["put_wait_s"] == pytest.approx(0.75)
    assert perf["store_s"] >= 0.0
    assert perf["forward_s"] >= 0.0

    other = object.__new__(runner_ext.LumenRLModelRunner)
    other.rank = 1
    other._extract_mode = True
    other._mooncake_store = SimpleNamespace(timings=TransferTimings())
    perf_path.unlink()
    assert other.forward(capture) == "forward"
    assert not perf_path.exists()
    assert other._mooncake_store.timings.snapshot() == {}


def test_actor_reports_call_time_and_idle_gap_without_dropping_engine_perf():
    class EngineStub:
        def extract_hidden_state_manifest(self, input_ids, attention_mask, row_segments=None):
            return {"mooncake_keys": ["row-0"], "teacher_perf": {"cmd_s": 1.0}}

    actor = object.__new__(AtomTeacherRayActor)
    actor._generate_mode = "prefill"
    actor.replica_index = 0
    actor._engine = EngineStub()
    batch = (torch.tensor([[1, 2]]), torch.tensor([[1, 1]]), torch.tensor([[0, 1]]))

    first = actor.process_prefill_batch(0, *batch)["teacher_perf"]
    second = actor.process_prefill_batch(1, *batch)["teacher_perf"]

    assert first["cmd_s"] == 1.0
    assert first["idle_s"] == 0.0
    assert second["idle_s"] >= 0.0
    assert second["end_time"] >= first["end_time"]
    assert first["actor_s"] >= 0.0


def test_teacher_perf_metrics_drop_wall_clock_and_convert_bytes():
    metrics = SpecDistillTrainer._teacher_perf_metrics({
        "teacher_perf": {
            "actor_s": 2.0, "end_time": 1.7e9, "put_bytes": 2 * 2**30, "rows": 16,
        }
    })

    assert metrics == {
        "teacher/actor_s": 2.0, "teacher/put_gib": 2.0, "teacher/rows": 16.0,
    }
    assert SpecDistillTrainer._teacher_perf_metrics({}) == {}


@pytest.mark.parametrize("finished_ago, teacher_wait_is_ray", [(-1e6, True), (1e6, False)])
def test_manifest_receive_splits_teacher_wait_from_transfer(finished_ago, teacher_wait_is_ray):
    end_time = time.time() - finished_ago

    class Ray:
        def get(self, ref):
            return {"mooncake_keys": ["k"], "teacher_perf": {"end_time": end_time}}

    trainer = object.__new__(SpecDistillTrainer)
    trainer._rank, trainer._is_distributed = 0, False
    trainer._ray = Ray()

    manifest = trainer._receive_disaggregated_manifest(object())

    timing = trainer._last_manifest_timing
    assert manifest["mooncake_keys"] == ["k"]
    assert timing["timing/manifest_ray_s"] >= 0.0
    # Clamped into [0, ray_s] whichever way the two wall clocks disagree.
    expected = timing["timing/manifest_ray_s"] if teacher_wait_is_ray else 0.0
    assert timing["timing/manifest_teacher_wait_s"] == expected


def test_dspark_step_accumulates_h2d_forward_backward_only_when_asked():
    hidden_dim, seq_len = 4, 3

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, aux_hidden_states, target_hidden_states, **kwargs):
            loss = self.scale * (aux_hidden_states.sum() + target_hidden_states.sum())
            zero = torch.zeros(())
            return {"total_loss": loss, "ce_loss": zero, "tv_loss": zero,
                    "conf_loss": zero, "losses": [], "accuracies": []}

    trainer = object.__new__(SpecDistillTrainer)
    trainer._device = torch.device("cpu")
    trainer._draft_model = Draft()
    trainer._lm_head_weight = torch.zeros(8, hidden_dim)
    trainer._embed_weight = torch.zeros(8, hidden_dim)
    trainer._norm_weight = None
    trainer.config = SimpleNamespace(algorithm=SimpleNamespace(spec_distill=SimpleNamespace(
        separate_last_hidden=False, ce_loss_alpha=1.0, l1_loss_alpha=0.0,
        confidence_loss_alpha=0.0, loss_decay_gamma=1.0,
    )))
    teacher = {
        "hidden_states": torch.ones(1, seq_len, 2 * hidden_dim),
        "input_ids": torch.ones(1, seq_len, dtype=torch.int64),
        "last_hidden_states": torch.ones(1, seq_len, hidden_dim),
    }
    mask = torch.ones(1, seq_len, dtype=torch.long)

    trainer._train_timings = {"h2d_s": 0.0, "forward_s": 0.0, "backward_s": 0.0}
    metrics = trainer._train_step_dspark(teacher, mask, mask)
    trainer._train_step_dspark(teacher, mask, mask)

    assert metrics["loss"] == float(seq_len * 3 * hidden_dim)
    assert trainer._draft_model.scale.grad is not None
    assert set(trainer._train_timings) == {"h2d_s", "forward_s", "backward_s"}
    assert all(value >= 0.0 for value in trainer._train_timings.values())

    trainer._train_timings = None
    assert trainer._train_step_dspark(teacher, mask, mask)["loss"] == metrics["loss"]


def test_max_reduce_replaces_only_the_named_keys(monkeypatch):
    trainer = object.__new__(SpecDistillTrainer)
    trainer._is_distributed = True
    trainer._device = torch.device("cpu")
    metrics = {"a": 1.0, "b_max": 2.0, "c": 3.0}
    calls = []

    def fake_all_reduce(values, op):
        calls.append(op)
        values.copy_(torch.maximum(values, torch.tensor([5.0, 0.0], dtype=torch.float64)))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    trainer._max_metrics_across_ranks(metrics, ["b_max", "c"])

    assert calls == [torch.distributed.ReduceOp.MAX]
    assert metrics == {"a": 1.0, "b_max": 5.0, "c": 3.0}


def test_fetch_perf_keys_are_fixed_and_count_this_ranks_rows(monkeypatch):
    monkeypatch.delenv("LUMENRL_BALANCE_DISAGG_ROWS", raising=False)
    hidden_dim = 4

    class Store:
        timings = TransferTimings()

        def get(self, key, shapes, dtypes, device):
            self.timings.add(get_s=0.5, get_calls=1, get_bytes=2**30, copy_s=0.25)
            seq_len = shapes["input_ids"][0]
            return SimpleNamespace(
                hidden_states=torch.zeros(shapes["hidden_states"]),
                input_ids=torch.zeros(seq_len, dtype=torch.int64),
                last_hidden_states=torch.zeros(shapes["last_hidden_states"]),
            )

        def remove_eagle3_tensors(self, key, **kwargs):
            self.timings.add(remove_s=0.125, remove_calls=1)

    store = Store()
    store.timings.add(get_s=99.0)  # left over from before this fetch
    trainer = object.__new__(SpecDistillTrainer)
    trainer._rank, trainer._world_size = 1, 2
    trainer._disagg_mooncake_store = store
    trainer._lm_head_weight = torch.zeros(8, hidden_dim)
    trainer._teacher_hidden_contract_logged = True
    trainer.config = SimpleNamespace(algorithm=SimpleNamespace(
        spec_distill=SimpleNamespace(aux_hidden_state_layer_ids=[1, 2], draft_type="eagle3"),
        teacher=SimpleNamespace(inference_backend="vllm"),
    ))
    manifest = {
        "mooncake_keys": ["k0", "k1", "k2", "k3"],
        "sequence_lengths": [3, 5, 2, 7],
        "attention_mask": torch.ones(4, 7, dtype=torch.long),
        "loss_mask": torch.ones(4, 7, dtype=torch.long),
        "hidden_dim": hidden_dim,
        "num_aux_layers": 2,
        "aux_layer_ids": [1, 2],
    }

    fetched, attention_mask, _, perf = trainer._load_disagg_rows(manifest)

    assert [int(row["input_ids"].shape[0]) for row in fetched] == [5, 7]
    assert attention_mask.shape == (2, 7)
    assert list(perf) == [
        "fetch/total_s", "fetch/get_s", "fetch/get_calls", "fetch/get_retry_sleep_s",
        "fetch/copy_s", "fetch/remove_s", "fetch/row_max_s", "fetch/rows",
        "fetch/tokens", "fetch/gib", "fetch/get_gib_s", "fetch/slot_fallback",
        "fetch/own_segment_gib",
    ]
    assert perf["fetch/get_s"] == 1.0
    assert perf["fetch/get_calls"] == 2.0
    assert perf["fetch/copy_s"] == 0.5
    assert perf["fetch/remove_s"] == 0.25
    assert perf["fetch/rows"] == 2.0
    assert perf["fetch/tokens"] == 12.0
    assert perf["fetch/gib"] == 2.0
    assert perf["fetch/get_gib_s"] == 2.0
    assert perf["fetch/get_retry_sleep_s"] == 0.0
    assert store.timings.snapshot() == {}
