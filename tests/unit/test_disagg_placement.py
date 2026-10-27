from __future__ import annotations

import ctypes
import sys
import threading
import types
from types import SimpleNamespace

import pytest
import torch

from lumenrl.engine.inference.atom_teacher_engine import AtomTeacherEngine
from lumenrl.engine.inference.atom_teacher_ray import AtomTeacherRayActor
from lumenrl.trainer.spec_distill_trainer import SpecDistillTrainer
from lumenrl.transfer.eagle_mooncake_store import (
    AsyncPutManager,
    EagleMooncakeStore,
    RingPutManager,
)


class _ReplicateConfig:
    """The fields mooncake.store.ReplicateConfig exposes (no with_hard_pin)."""

    def __init__(self):
        self.replica_num = 1
        self.with_soft_pin = False
        self.preferred_segments: list[str] = []
        self.preferred_segment = ""
        self.prefer_alloc_in_same_node = False


@pytest.fixture
def mooncake_module(monkeypatch):
    store_module = types.ModuleType("mooncake.store")
    store_module.ReplicateConfig = _ReplicateConfig
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    return store_module


class _PutClient:
    def __init__(self):
        self.puts: list[tuple[list[str], object]] = []

    def batch_put_from(self, keys, ptrs, sizes, config=None):
        self.puts.append((list(keys), config))
        return [0] * len(keys)


def _store(client=None, hostname="10.0.0.9:7001") -> EagleMooncakeStore:
    store = EagleMooncakeStore(SimpleNamespace(
        get_retry_max_wait_seconds=5.0, metadata_server="http://10.0.0.1:8080/metadata"))
    store._store = client if client is not None else SimpleNamespace()
    store._store.get_hostname = lambda: hostname
    store._initialized = True
    store._init_event.set()
    return store


def test_key_suffix_selects_a_cached_placement_config(mooncake_module):
    store = _store()

    config = store._config_for_key("atom_x3_5@10.0.0.2:7001")

    assert isinstance(config, _ReplicateConfig)
    assert config.preferred_segment == "10.0.0.2:7001"
    assert config.preferred_segments == ["10.0.0.2:7001"]
    assert config.replica_num == 1
    assert store._config_for_key("atom_x4_0@10.0.0.2:7001") is config
    assert store._config_for_key("atom_x3_5@10.0.0.3:7001").preferred_segment == "10.0.0.3:7001"
    assert store._config_for_key("atom_x3_5") is None
    assert store._config_for_key("atom_x3_5@") is None


def test_placement_overrides_a_base_that_names_its_own_segment(mooncake_module):
    store = _store()
    base = _ReplicateConfig()
    base.preferred_segment = "10.0.0.9:7001"
    base.preferred_segments = ["10.0.0.9:7001"]
    base.with_soft_pin = True
    base.prefer_alloc_in_same_node = True
    store._replicate_config = base

    config = store._config_for_key("k@10.0.0.2:7001")

    assert config.preferred_segment == "10.0.0.2:7001"
    assert config.preferred_segments == ["10.0.0.2:7001"]
    assert config.with_soft_pin and config.prefer_alloc_in_same_node
    assert base.preferred_segments == ["10.0.0.9:7001"]


def test_placement_without_the_deprecated_field(mooncake_module):
    class Config:
        def __init__(self):
            self.replica_num = 1
            self.preferred_segments: list[str] = []

    mooncake_module.ReplicateConfig = Config

    config = _store()._config_for_key("k@10.0.0.2:7001")

    assert config.preferred_segments == ["10.0.0.2:7001"]
    assert not hasattr(config, "preferred_segment")


def test_sync_put_sends_the_rows_placement(mooncake_module):
    client = _PutClient()
    store = _store(client)
    config = store._config_for_key("k@seg")

    store._do_sync_batch_put(["k@seg_hs", "k@seg_ids"], [1, 2], [8, 8], config=config)
    store._do_sync_batch_put(["j_hs"], [1], [8])

    assert client.puts == [(["k@seg_hs", "k@seg_ids"], config), (["j_hs"], None)]


@pytest.mark.parametrize("ring", [False, True])
def test_async_put_sends_the_rows_placement(ring):
    client = _PutClient()
    default, placed = object(), object()
    if ring:
        buffer = torch.zeros(64, dtype=torch.uint8)
        manager = RingPutManager(
            client, SimpleNamespace(ptr=buffer.data_ptr(), size=64), replicate_config=default)
        ptr = buffer.data_ptr()
    else:
        manager = AsyncPutManager(client, replicate_config=default)
        ptr = 4096

    manager.submit(["a_hs"], [ptr], [8], ptr, replicate_config=placed)
    manager.submit(["b_hs"], [ptr + 8], [8], ptr + 8)
    manager.shutdown()

    assert client.puts == [(["a_hs"], placed), (["b_hs"], default)]


def _replica(endpoint: str, memory: bool = True):
    buffer = SimpleNamespace(transport_endpoint=endpoint)
    return SimpleNamespace(
        is_memory_replica=lambda: memory,
        get_memory_descriptor=lambda: SimpleNamespace(buffer_descriptor=buffer),
    )


def test_own_segment_bytes_counts_first_replicas_in_this_segment():
    own = "10.0.0.9:7001"
    client = SimpleNamespace(batch_get_replica_desc=lambda keys: {
        "a": [_replica(own)],
        "b": [_replica("10.0.0.2:7001"), _replica(own)],
        "c": [],
        "d": [_replica(own, memory=False)],
        "e": [_replica(own)],
    })
    store = _store(client, hostname=own)

    assert store.own_segment_bytes(list("abcde"), [1, 2, 4, 8, 16]) == 17


def test_get_into_records_own_segment_bytes_only_when_asked():
    sizes = {"a": 16, "b": 32}
    own = "10.0.0.9:7001"

    def batch_get_into(keys, ptrs, lengths):
        for ptr, size in zip(ptrs, lengths, strict=True):
            ctypes.memset(ptr, 1, size)
        return list(lengths)

    client = SimpleNamespace(
        batch_get_into=batch_get_into,
        batch_get_replica_desc=lambda keys: {"a": [_replica(own)], "b": [_replica("x:1")]},
    )
    store = _store(client, hostname=own)
    buffer = torch.zeros(64, dtype=torch.uint8)
    ptrs = [buffer.data_ptr(), buffer.data_ptr() + 16]

    store.get_into(list(sizes), ptrs, list(sizes.values()))
    assert "own_segment_bytes" not in store.timings.pop()

    store.measure_own_segment = True
    store.get_into(list(sizes), ptrs, list(sizes.values()))
    assert store.timings.pop()["own_segment_bytes"] == 16

    # No segment name to compare with, so no figure rather than a wrong zero.
    store.config.metadata_server = "P2PHANDSHAKE"
    store.get_into(list(sizes), ptrs, list(sizes.values()))
    assert "own_segment_bytes" not in store.timings.pop()


@pytest.mark.parametrize("balance", ["0", "1"])
def test_submitted_rows_go_to_the_segment_of_the_rank_that_reads_them(monkeypatch, balance):
    monkeypatch.setenv("LUMENRL_BALANCE_DISAGG_ROWS", balance)
    width, world = 10, 4
    # The empty row checks the teacher worker's clamp to [1, width].
    mask_lengths = [0, 10, 3, 7, 1, 9, 4, 4, 2, 8, 6, 5, 10, 1, 3, 7]
    attention_mask = torch.zeros(len(mask_lengths), width, dtype=torch.long)
    for row, length in enumerate(mask_lengths):
        attention_mask[row, :length] = 1
    calls = []
    actor = SimpleNamespace(process_prefill_batch=SimpleNamespace(
        remote=lambda *args, **kwargs: calls.append((args, kwargs))))

    trainer = object.__new__(SpecDistillTrainer)
    trainer._world_size = world
    trainer._teacher_ray_actors = [actor]
    trainer._draft_segment_names = [f"seg{rank}" for rank in range(world)]
    trainer.config = SimpleNamespace(algorithm=SimpleNamespace(
        teacher=SimpleNamespace(generate_mode="prefill")))
    trainer._get_batch_sequences = lambda step: (
        torch.ones(len(mask_lengths), width, dtype=torch.long),
        attention_mask,
        attention_mask.clone(),
    )

    trainer._submit_disaggregated_teacher_batch(3)

    (_, kwargs), = calls
    segments = kwargs["row_segments"]
    # The lengths the manifest will carry: the teacher worker's max(1, min(n, T)).
    published = [max(1, min(length, width)) for length in mask_lengths]
    for rank in range(world):
        rows = trainer._default_disagg_rows(published, rank)
        assert len(rows) == len(mask_lengths) // world
        assert {segments[row] for row in rows} == {f"seg{rank}"}


def test_submit_without_placement_sends_no_segments():
    calls = []
    actor = SimpleNamespace(process_prefill_batch=SimpleNamespace(
        remote=lambda *args, **kwargs: calls.append(kwargs)))
    trainer = object.__new__(SpecDistillTrainer)
    trainer._world_size = 2
    trainer._teacher_ray_actors = [actor]
    trainer._draft_segment_names = None
    trainer.config = SimpleNamespace(algorithm=SimpleNamespace(
        teacher=SimpleNamespace(generate_mode="prefill")))
    mask = torch.ones(2, 4, dtype=torch.long)
    trainer._get_batch_sequences = lambda step: (mask, mask, mask)

    trainer._submit_disaggregated_teacher_batch(0)

    assert calls == [{"row_segments": None}]


@pytest.mark.parametrize("row_segments", [None, ["seg1", "seg0"]])
def test_actor_forwards_row_segments(row_segments):
    seen = []

    class EngineStub:
        def extract_hidden_state_manifest(self, input_ids, attention_mask, row_segments=None):
            seen.append(row_segments)
            return {"mooncake_keys": ["a", "b"]}

    actor = object.__new__(AtomTeacherRayActor)
    actor._generate_mode = "prefill"
    actor.replica_index = 0
    actor._engine = EngineStub()
    batch = torch.ones(2, 3, dtype=torch.long)

    actor.process_prefill_batch(1, batch, batch, batch, row_segments=row_segments)

    assert seen == [row_segments]


def _engine(tmp_path, seen):
    class Engine(AtomTeacherEngine):
        is_alive = True

        def switch_mode(self, mode):
            pass

        def _send_cmd_unlocked(self, cmd):
            seen.update(torch.load(cmd["input_path"], weights_only=True))
            return {"status": "ok", "mooncake_keys": ["a@s1", "b@s0"],
                    "seq_lens": {"a@s1": 3, "b@s0": 1}, "T": 3, "D": 4}

        def shutdown(self):
            # No worker was started; __del__ calls this.
            pass

    engine = object.__new__(Engine)
    engine._proc = None
    engine._transport = "mooncake"
    engine._cmd_lock = threading.Lock()
    engine._req_counter = 0
    engine._hidden_dir = str(tmp_path)
    engine._aux_layer_indices = [1, 2]
    engine._atom_config = {}
    return engine


def test_engine_hands_row_segments_to_its_worker(tmp_path):
    seen = {}
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]])

    manifest = _engine(tmp_path, seen).extract_hidden_state_manifest(
        torch.ones(2, 3, dtype=torch.long), mask, row_segments=["s1", "s0"])

    assert seen["row_segments"] == ["s1", "s0"]
    assert seen["lengths"].tolist() == [3, 1]
    assert manifest["mooncake_keys"] == ["a@s1", "b@s0"]
    assert manifest["sequence_lengths"] == [3, 1]

    seen.clear()
    _engine(tmp_path, seen).extract_hidden_state_manifest(torch.ones(2, 3, dtype=torch.long), mask)
    assert "row_segments" not in seen


def test_engine_rejects_a_segment_list_of_the_wrong_length(tmp_path):
    with pytest.raises(ValueError, match="1 row segments for a batch of 2 rows"):
        _engine(tmp_path, {}).extract_hidden_state_manifest(
            torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.long),
            row_segments=["s0"])
