from __future__ import annotations

import itertools
import sys
import threading
import types
from concurrent.futures import wait
from types import SimpleNamespace

import pytest

from lumenrl.transfer import eagle_mooncake_store
from lumenrl.transfer.eagle_mooncake_store import (
    AsyncPutManager,
    EagleMooncakeStore,
    RingPutManager,
    TransferTimings,
)


class _GatedMooncake:
    """Each row's PUT blocks until the test opens that row's gate."""

    def __init__(self):
        self._gates: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self.done: list[tuple[str, int, int]] = []

    def gate(self, key: str) -> threading.Event:
        with self._lock:
            return self._gates.setdefault(key, threading.Event())

    def batch_put_from(self, keys, ptrs, sizes, config=None):
        assert self.gate(keys[0]).wait(timeout=10), f"gate for {keys[0]} never opened"
        self.done.append((keys[0], ptrs[0], sum(sizes)))
        return [0] * len(keys)


def _ring(size: int):
    return SimpleNamespace(size=size, ptr=4096)


def _put(manager, ring, key: str, nbytes: int) -> int:
    offset = manager.acquire(ring, nbytes)
    manager.submit([key], [ring.ptr + offset], [nbytes], ring.ptr)
    return offset


def test_ring_appends_wraps_and_waits_only_for_the_oldest_rows():
    mooncake, ring, timings = _GatedMooncake(), _ring(100), TransferTimings()
    manager = RingPutManager(mooncake, ring, timings=timings)
    try:
        assert _put(manager, ring, "r1", 40) == 0
        assert _put(manager, ring, "r2", 40) == 40
        assert timings.snapshot()["put_ring_peak_bytes"] == 80

        # r3 does not fit in [80, 100), wraps to 0 and so needs r1's space.
        offsets = []
        third = threading.Thread(target=lambda: offsets.append(_put(manager, ring, "r3", 40)))
        third.start()
        third.join(timeout=0.2)
        assert third.is_alive()
        mooncake.gate("r1").set()
        third.join(timeout=10)
        assert offsets == [0]
        # r2 is still in flight: r3 waited for r1 alone.
        assert not mooncake.gate("r2").is_set()
        # r2, the skipped tail [80, 100) and r3 fill the whole ring.
        assert timings.snapshot()["put_ring_peak_bytes"] == 100

        mooncake.gate("r2").set()
        mooncake.gate("r3").set()
        manager.drain()
        manager.check_last_error()
    finally:
        for key in ("r1", "r2", "r3"):
            mooncake.gate(key).set()
        manager.shutdown()

    assert mooncake.done == [("r1", 4096, 40), ("r2", 4136, 40), ("r3", 4096, 40)]
    perf = timings.snapshot()
    assert perf["put_lock_wait_s"] >= 0.0 and perf["put_rpc_s"] >= 0.0


def test_ring_reuses_space_of_finished_rows_without_waiting():
    mooncake, ring = _GatedMooncake(), _ring(100)
    manager = RingPutManager(mooncake, ring, timings=TransferTimings())
    try:
        for key in ("r1", "r2", "r3"):
            mooncake.gate(key).set()
        assert _put(manager, ring, "r1", 60) == 0
        manager.drain()
        # Only 40 bytes are left before the end, so r2 restarts at 0.
        assert _put(manager, ring, "r2", 60) == 0
        assert _put(manager, ring, "r3", 40) == 60
        manager.drain()
    finally:
        manager.shutdown()
    assert [key for key, _, _ in mooncake.done] == ["r1", "r2", "r3"]


def test_ring_rejects_a_row_larger_than_the_ring_and_a_foreign_buffer():
    ring = _ring(100)
    manager = RingPutManager(_GatedMooncake(), ring)
    try:
        with pytest.raises(ValueError, match="Buffer overflow"):
            manager.acquire(ring, 101)
        with pytest.raises(ValueError, match="own ring"):
            manager.acquire(_ring(100), 10)
    finally:
        manager.shutdown()


def test_ring_put_error_surfaces_on_next_acquire_and_on_flush():
    class Failing:
        def batch_put_from(self, keys, ptrs, sizes, config=None):
            raise RuntimeError(f"transfer failed for {keys[0]}")

    ring = _ring(100)
    manager = RingPutManager(Failing(), ring)
    try:
        _put(manager, ring, "r1", 10)
        wait([manager._rows[0][2]])
        with pytest.raises(RuntimeError, match="r1"):
            manager.acquire(ring, 10)

        _put(manager, ring, "r2", 10)
        manager.drain()
        with pytest.raises(RuntimeError):
            manager.check_last_error()
    finally:
        manager.shutdown()


def test_buffer_pool_put_lock_wait_is_not_counted_as_rpc_time():
    class Slow:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def batch_put_from(self, keys, ptrs, sizes):
            if keys[0] == "first":
                self.started.set()
                assert self.release.wait(timeout=10)
            return [0] * len(keys)

    mooncake, timings = Slow(), TransferTimings()
    manager = AsyncPutManager(mooncake, max_workers=2, timings=timings)
    try:
        manager.submit(["first"], [0], [16], owner_buffer_ptr=1)
        assert mooncake.started.wait(timeout=10)
        manager.submit(["second"], [0], [16], owner_buffer_ptr=2)
        threading.Event().wait(0.5)
        mooncake.release.set()
        manager.wait_for_buffer(1)
        manager.wait_for_buffer(2)
    finally:
        manager.shutdown()

    perf = timings.snapshot()
    # The second PUT queued ~0.5 s behind the first one's lock. Counting that
    # as RPC time would report ~1.0 s of RPC for ~0.5 s of transfer.
    assert perf["put_lock_wait_s"] >= 0.4
    assert perf["put_rpc_s"] < 0.8


def test_peak_keeps_the_maximum():
    timings = TransferTimings()
    timings.peak(put_ring_peak_bytes=5)
    timings.peak(put_ring_peak_bytes=3)
    timings.add(put_rows=1)
    assert timings.snapshot() == {"put_ring_peak_bytes": 5.0, "put_rows": 1.0}


@pytest.mark.parametrize("ring_env, manager_type, buffers", [
    ("1", RingPutManager, [2 * 64]),
    ("0", AsyncPutManager, [64, 64]),
])
def test_setup_selects_ring_by_env(monkeypatch, ring_env, manager_type, buffers):
    registered = []
    addresses = itertools.count(1)

    class Store:
        def setup(self, **kwargs):
            return 0

        def register_buffer(self, ptr, size):
            registered.append(size)
            return 0

        def close(self):
            pass

    class FakeHostBuffer:
        def __init__(self, size):
            self.size, self.ptr = size, 4096 * next(addresses)

        def free(self):
            pass

    store_module = types.ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = Store
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    monkeypatch.setattr(eagle_mooncake_store, "HostBuffer", FakeHostBuffer)
    monkeypatch.setattr(eagle_mooncake_store.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv(eagle_mooncake_store.RING_PUT_ENV, ring_env)

    store = EagleMooncakeStore(SimpleNamespace(
        global_segment_size=1 << 20, local_buffer_size=1 << 20, local_hostname="h",
        metadata_server="", master_server_address="", protocol="tcp", device_name="",
        enable_hard_pin=False, async_put_pool_size=2, host_buffer_size=64,
        enable_gpu_direct=False,
    ))
    store.setup()
    try:
        assert type(store._async_put_manager) is manager_type
        assert [buf.size for buf in store._host_buffer_pool._buffers] == buffers
        assert registered == buffers
    finally:
        store.close()
