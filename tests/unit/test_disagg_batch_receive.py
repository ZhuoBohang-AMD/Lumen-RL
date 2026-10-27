from __future__ import annotations

import ctypes
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from lumenrl.trainer.spec_distill_trainer import (
    SpecDistillTrainer,
    _DisaggReceiveSlots,
    _read_disagg_rows_into_slot,
)
from lumenrl.transfer import eagle_mooncake_store
from lumenrl.transfer.eagle_mooncake_store import EagleMooncakeStore

HIDDEN_DIM = 4
NUM_AUX = 2


def _payload(key: str, size: int) -> torch.Tensor:
    seed = sum(key.encode()) % 251
    return (torch.arange(size, dtype=torch.int64) * 7 + seed).remainder(256).to(torch.uint8)


class _Client:
    """A Mooncake client whose keys hold ``_payload`` bytes once their put lands."""

    def __init__(self, sizes: dict[str, int], codes=None):
        self.sizes = sizes
        # key -> list of codes to return before the real read, e.g. [-704, -703].
        self.codes = codes if codes is not None else {}
        self.calls: list[list[str]] = []

    def batch_get_into(self, keys, ptrs, sizes):
        self.calls.append(list(keys))
        results = []
        for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
            queued = self.codes.get(key)
            code = queued.pop(0) if queued else None
            if code is not None:
                results.append(code)
                continue
            data = _payload(key, self.sizes[key])
            if size < data.numel():
                results.append(-600)
                continue
            ctypes.memmove(ptr, data.data_ptr(), data.numel())
            results.append(data.numel())
        return results


def _store(client, max_wait=5.0) -> EagleMooncakeStore:
    store = EagleMooncakeStore(SimpleNamespace(get_retry_max_wait_seconds=max_wait))
    store._store = client
    store._initialized = True
    store._init_event.set()
    return store


def _read(store, keys, sizes):
    buffer = torch.zeros(sum(sizes) + 64 * len(keys), dtype=torch.uint8)
    offsets, offset = [], 0
    for size in sizes:
        offsets.append(offset)
        offset += size + 64
    store.get_into(keys, [buffer.data_ptr() + o for o in offsets], sizes)
    return [buffer[o:o + size] for o, size in zip(offsets, sizes, strict=True)]


def test_get_into_reads_every_key_in_one_call():
    sizes = {f"k{i}": 1000 + 37 * i for i in range(8)}
    client = _Client(sizes)
    store = _store(client)
    keys = list(sizes)

    received = _read(store, keys, [sizes[k] for k in keys])

    for key, data in zip(keys, received, strict=True):
        assert torch.equal(data, _payload(key, sizes[key]))
    assert client.calls == [keys]
    perf = store.timings.pop()
    assert perf["get_calls"] == 1
    assert perf["get_bytes"] == sum(sizes.values())
    assert perf["get_retry_sleep_s"] == 0.0


def test_get_into_retries_only_keys_not_landed_yet():
    sizes = {"a": 64, "b": 128, "c": 32}
    client = _Client(sizes, codes={"b": [-704, -703]})
    store = _store(client)

    received = _read(store, list(sizes), list(sizes.values()))

    assert all(torch.equal(d, _payload(k, sizes[k])) for k, d in zip(sizes, received))
    assert client.calls == [["a", "b", "c"], ["b"], ["b"]]
    perf = store.timings.pop()
    assert perf["get_calls"] == 3
    assert perf["get_retry_sleep_s"] > 0


@pytest.mark.parametrize("code", [-600, -1, 0, 63])
def test_get_into_raises_on_any_other_result(code):
    sizes = {"a": 64, "b": 64}
    store = _store(_Client(sizes, codes={"b": [code]}))
    with pytest.raises(RuntimeError, match=f"b={code}"):
        _read(store, list(sizes), list(sizes.values()))


def test_get_into_times_out_on_a_key_that_never_lands():
    sizes = {"a": 64}
    store = _store(_Client(sizes, codes={"a": [-703] * 100}), max_wait=0.12)
    with pytest.raises(RuntimeError, match="timed out"):
        _read(store, ["a"], [64])


class _SlotStore:
    """get_into/remove_keys over ``_Client``; records what was removed."""

    def __init__(self, sizes, codes=None):
        self.client = _Client(sizes, codes)
        self.inner = _store(self.client)
        self.removed: list[str] = []
        self.timings = self.inner.timings

    def get_into(self, *args, **kwargs):
        return self.inner.get_into(*args, **kwargs)

    def remove_keys(self, keys, label=None):
        self.removed.extend(keys)
        self.timings.add(remove_calls=1, remove_s=0.125)


def _row_sizes(base: str, seq_len: int) -> dict[str, int]:
    return {
        f"{base}_hs": seq_len * NUM_AUX * HIDDEN_DIM * 2,
        f"{base}_ids": seq_len * 8,
        f"{base}_lhs": seq_len * HIDDEN_DIM * 2,
    }


def _expected(base: str, name: str, seq_len: int) -> torch.Tensor:
    suffix, dtype, tail = {
        "hidden_states": ("_hs", torch.bfloat16, (NUM_AUX * HIDDEN_DIM,)),
        "input_ids": ("_ids", torch.int64, ()),
        "last_hidden_states": ("_lhs", torch.bfloat16, (HIDDEN_DIM,)),
    }[name]
    key = f"{base}{suffix}"
    size = _row_sizes(base, seq_len)[key]
    return _payload(key, size).view(dtype).view(seq_len, *tail)


def test_rows_are_views_of_the_slot_with_their_own_shapes():
    seq_lens = [3, 5, 1]
    keys = ["r0", "r1", "r2"]
    sizes = {k: v for key, n in zip(keys, seq_lens) for k, v in _row_sizes(key, n).items()}
    store = _SlotStore(sizes)
    slots = _Slots(4096)
    slot = slots.slots[0]

    rows = _read_disagg_rows_into_slot(store, slots, 0, keys, seq_lens, NUM_AUX * HIDDEN_DIM,
                                       HIDDEN_DIM)

    assert rows is not None and len(rows) == 3
    for key, seq_len, row in zip(keys, seq_lens, rows, strict=True):
        assert set(row) == {"hidden_states", "input_ids", "last_hidden_states"}
        for name, tensor in row.items():
            assert tensor.untyped_storage().data_ptr() == slot.untyped_storage().data_ptr()
            assert (tensor.data_ptr() - slot.data_ptr()) % 256 == 0
            assert torch.equal(tensor, _expected(key, name, seq_len))
    assert sorted(store.removed) == sorted(sizes)


def test_rows_that_do_not_fit_are_not_read_or_removed():
    keys, seq_lens = ["r0"], [100]
    store = _SlotStore(_row_sizes("r0", 100))
    rows = _read_disagg_rows_into_slot(store, _Slots(1024), 0, keys,
                                       seq_lens, NUM_AUX * HIDDEN_DIM, HIDDEN_DIM)
    assert rows is None
    assert store.client.calls == [] and store.removed == []


def test_keys_are_removed_when_the_read_fails():
    sizes = _row_sizes("r0", 2)
    store = _SlotStore(sizes, codes={"r0_lhs": [-800]})
    with pytest.raises(RuntimeError, match="r0_lhs=-800"):
        _read_disagg_rows_into_slot(store, _Slots(1024), 0, ["r0"], [2],
                                    NUM_AUX * HIDDEN_DIM, HIDDEN_DIM)
    assert sorted(store.removed) == sorted(sizes)


class _Slots:
    """Two fixed slots that never grow."""

    def __init__(self, size):
        self.slots = [torch.zeros(size, dtype=torch.uint8) for _ in range(2)]
        self.acquired: list[int] = []

    def acquire(self, step, nbytes):
        self.acquired.append(step)
        slot = self.slots[step % 2]
        return slot if nbytes <= slot.numel() else None


def _trainer(store, slots):
    trainer = object.__new__(SpecDistillTrainer)
    trainer._rank, trainer._world_size = 1, 2
    trainer._disagg_mooncake_store = store
    trainer._disagg_receive_slots = slots
    trainer._lm_head_weight = torch.zeros(8, HIDDEN_DIM)
    trainer._teacher_hidden_contract_logged = True
    trainer.config = SimpleNamespace(algorithm=SimpleNamespace(
        spec_distill=SimpleNamespace(aux_hidden_state_layer_ids=[1, 2], draft_type="eagle3"),
        teacher=SimpleNamespace(inference_backend="vllm"),
    ))
    return trainer


MANIFEST_KEYS = ["k0", "k1", "k2", "k3"]
MANIFEST_LENS = [3, 5, 2, 7]
PERF_KEYS = [
    "fetch/total_s", "fetch/get_s", "fetch/get_calls", "fetch/get_retry_sleep_s",
    "fetch/copy_s", "fetch/remove_s", "fetch/row_max_s", "fetch/rows",
    "fetch/tokens", "fetch/gib", "fetch/get_gib_s", "fetch/slot_fallback",
    "fetch/own_segment_gib",
]


def _manifest():
    return {
        "mooncake_keys": MANIFEST_KEYS,
        "sequence_lengths": MANIFEST_LENS,
        "attention_mask": torch.ones(4, 7, dtype=torch.long),
        "loss_mask": torch.ones(4, 7, dtype=torch.long),
        "hidden_dim": HIDDEN_DIM,
        "num_aux_layers": NUM_AUX,
        "aux_layer_ids": [1, 2],
    }


def test_trainer_fetch_reads_its_rows_into_the_step_slot():
    sizes = {k: v for key, n in zip(MANIFEST_KEYS, MANIFEST_LENS)
             for k, v in _row_sizes(key, n).items()}
    store = _SlotStore(sizes)
    slots = _Slots(8192)
    trainer = _trainer(store, slots)

    fetched, attention_mask, _, perf = trainer._load_disagg_rows(_manifest(), step=7)

    assert slots.acquired == [7]
    assert [int(row["input_ids"].shape[0]) for row in fetched] == [5, 7]
    for key, row in zip(["k1", "k3"], fetched, strict=True):
        for name, tensor in row.items():
            assert tensor.untyped_storage().data_ptr() == slots.slots[1].untyped_storage().data_ptr()
            assert torch.equal(tensor, _expected(key, name, int(row["input_ids"].shape[0])))
    assert attention_mask.shape == (2, 7)
    assert list(perf) == PERF_KEYS
    assert perf["fetch/slot_fallback"] == 0.0
    assert perf["fetch/get_calls"] == 1.0
    assert perf["fetch/remove_s"] == 0.125
    assert perf["fetch/gib"] * 2**30 == sum(v for k, v in sizes.items() if k[:2] in ("k1", "k3"))
    assert sorted(store.removed) == sorted(k for k in sizes if k[:2] in ("k1", "k3"))
    assert store.timings.snapshot() == {}


def test_trainer_fetch_falls_back_per_row_when_the_slot_is_too_small():
    class Store(_SlotStore):
        def get(self, key, shapes, dtypes, device):
            self.timings.add(get_s=0.5, get_calls=1, get_bytes=2**30, copy_s=0.25)
            return SimpleNamespace(
                hidden_states=torch.zeros(shapes["hidden_states"]),
                input_ids=torch.zeros(shapes["input_ids"], dtype=torch.int64),
                last_hidden_states=torch.zeros(shapes["last_hidden_states"]),
            )

        def remove_eagle3_tensors(self, key, **kwargs):
            self.removed.append(key)

    store = Store({})
    trainer = _trainer(store, _Slots(64))

    fetched, _, _, perf = trainer._load_disagg_rows(_manifest(), step=4)

    assert [int(row["input_ids"].shape[0]) for row in fetched] == [5, 7]
    assert store.removed == ["k1", "k3"]
    assert list(perf) == PERF_KEYS
    assert perf["fetch/slot_fallback"] == 1.0
    assert perf["fetch/get_calls"] == 2.0


def test_trainer_fetch_warns_when_rows_miss_its_own_segment(caplog):
    sizes = {k: v for key, n in zip(MANIFEST_KEYS, MANIFEST_LENS)
             for k, v in _row_sizes(key, n).items()}

    class Store(_SlotStore):
        own_fraction = 1.0
        config = SimpleNamespace(global_segment_size="64GB")

        def get_into(self, keys, ptrs, sizes):
            self.inner.get_into(keys, ptrs, sizes)
            self.timings.add(own_segment_bytes=self.own_fraction * sum(sizes))

    store = Store(sizes)
    trainer = _trainer(store, _Slots(8192))

    def fetch(step, own_fraction):
        store.own_fraction = own_fraction
        caplog.clear()
        perf = trainer._load_disagg_rows(_manifest(), step=step)[3]
        return perf, [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]

    perf, warnings = fetch(0, 1.0)
    assert perf["fetch/own_segment_gib"] == perf["fetch/gib"] and warnings == []

    perf, warnings = fetch(1, 0.5)
    assert len(warnings) == 1
    assert "50.0% of the bytes read" in warnings[0] and "(now 64GB)" in warnings[0]

    # Rate-limited: the next misplaced steps stay quiet until the 100th.
    assert all(fetch(step, 0.5)[1] == [] for step in range(2, 100))
    assert len(fetch(100, 0.5)[1]) == 1


def test_eval_fetch_without_a_step_keeps_the_per_row_path():
    class Store(_SlotStore):
        def get(self, key, shapes, dtypes, device):
            return SimpleNamespace(
                hidden_states=torch.zeros(shapes["hidden_states"]),
                input_ids=torch.zeros(shapes["input_ids"], dtype=torch.int64),
                last_hidden_states=torch.zeros(shapes["last_hidden_states"]),
            )

        def remove_eagle3_tensors(self, key, **kwargs):
            self.removed.append(key)

    slots = _Slots(8192)
    store = Store({})
    fetched, _, _, perf = _trainer(store, slots)._load_disagg_rows(_manifest(), rows=[0, 1])
    assert slots.acquired == [] and store.removed == ["k0", "k1"]
    assert perf["fetch/slot_fallback"] == 0.0


class _RegisteringStore:
    def __init__(self):
        self.registered: dict[int, int] = {}
        self.unregistered: list[int] = []

    def register_read_buffer(self, ptr, size):
        self.registered[ptr] = size

    def unregister_read_buffer(self, ptr):
        self.unregistered.append(ptr)
        del self.registered[ptr]


needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="pinned memory needs a GPU runtime")
# Above glibc's largest mmap threshold, so every slot has pages of its own and
# registering one never touches another's, as with real GiB-sized slots.
SLOT = 64 << 20


@needs_gpu
def test_receive_slots_alternate_and_refuse_a_held_slot():
    store = _RegisteringStore()
    slots = _DisaggReceiveSlots(store, SLOT, SLOT)
    assert sorted(store.registered.values()) == [SLOT, SLOT]

    first, second = slots.acquire(10, SLOT), slots.acquire(11, 1)
    assert first.is_pinned() and first.data_ptr() != second.data_ptr()
    assert {first.data_ptr(), second.data_ptr()} == set(store.registered)
    with pytest.raises(RuntimeError, match="still held by step 10"):
        slots.acquire(12, 1)
    slots.release(10)
    assert slots.acquire(12, 1).data_ptr() == first.data_ptr()
    slots.release(11)
    slots.release(9)  # stale: slot 1 is free already, so this is a no-op
    assert slots.acquire(13, 1).data_ptr() == second.data_ptr()
    slots.free()


@needs_gpu
def test_receive_slot_grows_to_a_power_of_two_and_drops_the_old_one():
    store = _RegisteringStore()
    slots = _DisaggReceiveSlots(store, SLOT, 4 * SLOT)
    held = slots.acquire(1, SLOT)
    old_ptr = slots._slots[0].data_ptr()

    grown = slots.acquire(2, SLOT + 1)

    assert grown.numel() == 2 * SLOT and grown.is_pinned()
    assert store.unregistered == [old_ptr]
    assert store.registered == {held.data_ptr(): SLOT, grown.data_ptr(): 2 * SLOT}
    assert slots._slots[1] is held
    slots.release(2)
    assert slots.acquire(4, 2 * SLOT) is grown
    slots.free()


@needs_gpu
def test_receive_slot_stays_put_above_its_maximum():
    store = _RegisteringStore()
    slots = _DisaggReceiveSlots(store, SLOT, SLOT + SLOT // 2)

    assert slots.acquire(0, 2 * SLOT) is None
    assert store.unregistered == []
    slots.release(0)
    # Capped at the maximum rather than the power of two above it.
    assert slots.acquire(2, SLOT + 1).numel() == SLOT + SLOT // 2
    slots.free()


@needs_gpu
def test_receive_slot_that_cannot_grow_keeps_its_size_and_stops_trying(monkeypatch):
    store = _RegisteringStore()
    slots = _DisaggReceiveSlots(store, SLOT, 4 * SLOT)
    attempts = []

    def fail(size):
        attempts.append(size)
        raise RuntimeError("out of pinned memory")

    monkeypatch.setattr(slots, "_allocate", fail)

    assert slots.acquire(0, SLOT + 1) is None
    slots.release(0)
    assert slots.acquire(2, SLOT + 1) is None
    assert attempts == [2 * SLOT]
    assert slots.max_size == SLOT and store.unregistered == []
    slots.release(2)
    assert slots.acquire(4, SLOT).numel() == SLOT
    monkeypatch.undo()
    slots.free()


def test_receive_slot_maximum_below_its_size_is_rejected():
    with pytest.raises(ValueError, match="above its maximum"):
        _DisaggReceiveSlots(_RegisteringStore(), 2 * SLOT, SLOT)


def test_read_only_store_has_no_put_pool_and_unregisters_read_buffers(monkeypatch):
    class Store:
        def __init__(self):
            self.registered, self.unregistered = [], []

        def setup(self, **kwargs):
            return 0

        def register_buffer(self, ptr, size):
            self.registered.append((ptr, size))
            return 0

        def unregister_buffer(self, ptr):
            self.unregistered.append(ptr)
            return 0

        def close(self):
            self.closed = True

    store_module = types.ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = Store
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    monkeypatch.setattr(eagle_mooncake_store.torch.cuda, "is_available", lambda: False)

    store = EagleMooncakeStore(SimpleNamespace(
        global_segment_size=1 << 20, local_buffer_size=1 << 20, local_hostname="h",
        metadata_server="", master_server_address="", protocol="tcp", device_name="",
        enable_hard_pin=False, async_put_pool_size=0, host_buffer_size=64,
        enable_gpu_direct=False,
    ))
    store.setup()
    assert store._host_buffer_pool is None and store._async_put_manager is None

    client = store._store
    store.register_read_buffer(4096, 100)
    store.register_read_buffer(8192, 200)
    assert client.registered == [(4096, 100), (8192, 200)]

    store.unregister_read_buffer(4096)
    assert client.unregistered == [4096]
    # A grown slot may come back at a freed one's address, with a new size.
    store.register_read_buffer(4096, 300)
    assert client.registered[-1] == (4096, 300)
    store.close()
    assert client.closed
