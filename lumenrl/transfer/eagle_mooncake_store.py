"""Eagle3 Mooncake store for speculative decoding hidden state transfer.

Eagle3 Mooncake store. Provides async RDMA/TCP
transfers of hidden states between SGLang inference and FSDP2 training.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

HIDDEN_STATES_STORAGE_DTYPE = torch.bfloat16

# "1" stages producer puts in one pinned ring with a single PUT thread instead of
# one row per buffer; see RingPutManager.
RING_PUT_ENV = "LUMENRL_MOONCAKE_RING_PUT"

# batch_get_into codes for a key whose put has not landed yet:
# REPLICA_IS_NOT_READY (put in flight) and OBJECT_NOT_FOUND (not started).
GET_NOT_READY_CODES = (-703, -704)
GET_INTO_RETRY_SECONDS = 0.05

# A put key "<base>@<segment>" asks the master to place the row in that segment
# (a store's segment_name()). The suffix stays part of the key for the reader.
SEGMENT_KEY_SEP = "@"

_DTYPE_ELEMENT_SIZES = {
    torch.float64: 8, torch.float32: 4, torch.bfloat16: 2, torch.float16: 2,
    torch.int64: 8, torch.int32: 4, torch.int16: 2, torch.int8: 1,
    torch.uint8: 1, torch.bool: 1,
}


def calculate_eagle3_buffer_size(
    max_seq_len: int,
    batch_size: int,
    hidden_dim: int,
    num_aux_layers: int = 3,
    include_last_hidden_states: bool = True,
    safety_margin: float = 1.1,
) -> int:
    bfloat16_size = 2
    int64_size = 8
    total = (batch_size * max_seq_len * hidden_dim * num_aux_layers * bfloat16_size
             + batch_size * max_seq_len * int64_size)
    if include_last_hidden_states:
        total += batch_size * max_seq_len * hidden_dim * bfloat16_size
    aligned = (int(total * safety_margin) + 255) // 256 * 256
    return aligned


def _format_bytes(size: int) -> str:
    if size < 0:
        return f"{size}B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024


class TransferTimings:
    """Thread-safe cumulative counters for Mooncake transfer timing.

    The producer's async put thread and the forward thread update the same
    counters, so every access takes the lock. Readers either take a cumulative
    ``snapshot()`` (teacher runner, which reports deltas per batch) or ``pop()``
    the counters accumulated since the previous pop (draft fetch, per step).
    ``peak()`` keys hold a running maximum instead of a sum and are named
    ``*_peak_bytes`` so per-batch delta reporting can pass them through as-is.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: Dict[str, float] = {}

    def add(self, **values: float) -> None:
        with self._lock:
            for key, value in values.items():
                self._values[key] = self._values.get(key, 0.0) + float(value)

    def peak(self, **values: float) -> None:
        with self._lock:
            for key, value in values.items():
                self._values[key] = max(self._values.get(key, 0.0), float(value))

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._values)

    def pop(self) -> Dict[str, float]:
        with self._lock:
            values, self._values = self._values, {}
        return values


# ---------------------------------------------------------------------------
# Host / GPU buffer classes
# ---------------------------------------------------------------------------

class HostBuffer:
    def __init__(self, size: int):
        self.size = size
        self._tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        self._ptr = self._tensor.data_ptr()

    @property
    def ptr(self) -> int:
        return self._ptr

    def copy_from_tensor(self, tensor: torch.Tensor, offset: int = 0) -> int:
        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if offset + nbytes > self.size:
            raise ValueError(f"Buffer overflow: need {offset + nbytes}, have {self.size}")
        self._tensor[offset:offset + nbytes].copy_(tensor.view(torch.uint8).view(-1))
        return nbytes

    def free(self) -> None:
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            self._ptr = 0


class HostBufferPool:
    def __init__(self, buffer_size: int, pool_size: int = 2, contiguous: bool = False):
        self.buffer_size = buffer_size
        self.pool_size = pool_size
        self.contiguous = contiguous
        self._buffers: List[HostBuffer] = []
        self._current_idx = 0

    def initialize(self) -> None:
        if self.contiguous:
            self._buffers.append(HostBuffer(self.buffer_size * self.pool_size))
            logger.info("Host buffer ring: %.1fGB (%d x %.1fGB)",
                        self.buffer_size * self.pool_size / 1024**3,
                        self.pool_size, self.buffer_size / 1024**3)
            return
        for _ in range(self.pool_size):
            self._buffers.append(HostBuffer(self.buffer_size))
        logger.info("Host buffer pool: %d x %.1fGB", self.pool_size, self.buffer_size / 1024**3)

    def get_buffer(self) -> HostBuffer:
        if not self._buffers:
            self.initialize()
        buf = self._buffers[self._current_idx]
        self._current_idx = (self._current_idx + 1) % len(self._buffers)
        return buf

    def shutdown(self) -> None:
        for buf in self._buffers:
            buf.free()
        self._buffers.clear()


class AsyncPutManager:
    def __init__(self, store: Any, max_workers: int = 1, replicate_config: Any = None,
                 timings: Optional[TransferTimings] = None):
        self._store = store
        self._replicate_config = replicate_config
        self._timings = timings
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="async_put")
        self._in_flight: Dict[int, Future] = {}
        self._last_error: Optional[BaseException] = None
        self._put_lock = threading.Lock()
        self._flight_lock = threading.Lock()

    def check_last_error(self) -> None:
        if self._last_error is not None:
            err = self._last_error
            self._last_error = None
            raise err

    def wait_for_buffer(self, buffer_ptr: int) -> None:
        with self._flight_lock:
            future = self._in_flight.pop(buffer_ptr, None)
        if future is None:
            return
        try:
            future.result()
        except Exception as exc:
            self._last_error = exc
            raise

    def acquire(self, buf: HostBuffer, nbytes: int) -> int:
        """Wait until *buf* may be overwritten; return the offset to stage a row at."""
        self.wait_for_buffer(buf.ptr)
        return 0

    def submit(self, keys, buffer_ptrs, sizes, owner_buffer_ptr,
               wait_event=None, device_index=None, replicate_config=None):
        future = self._executor.submit(
            self._do_put, keys, buffer_ptrs, sizes, wait_event, device_index,
            replicate_config=replicate_config,
        )
        with self._flight_lock:
            self._in_flight[owner_buffer_ptr] = future

    def _do_put(self, keys, buffer_ptrs, sizes, wait_event=None, device_index=None,
                _max_retries=3, _retry_delay=0.5, replicate_config=None):
        config = replicate_config if replicate_config is not None else self._replicate_config
        start = time.monotonic()
        if wait_event is not None:
            if device_index is not None:
                torch.cuda.set_device(device_index)
            wait_event.synchronize()
        staged = time.monotonic()
        lock_wait_s = 0.0
        try:
            for attempt in range(_max_retries):
                lock_start = time.monotonic()
                with self._put_lock:
                    lock_wait_s += time.monotonic() - lock_start
                    if config is not None:
                        results = self._store.batch_put_from(
                            keys, buffer_ptrs, sizes, config=config)
                    else:
                        results = self._store.batch_put_from(keys, buffer_ptrs, sizes)
                failures = [(k, r) for k, r in zip(keys, results) if r != 0]
                if not failures:
                    if attempt > 0:
                        logger.info("batch_put_from succeeded on attempt %d for %s",
                                    attempt + 1, keys[0])
                    return
                try:
                    self._store.batch_remove(keys, force=True)
                except Exception:
                    pass
                if attempt < _max_retries - 1:
                    detail = ", ".join(f"{k} (code={r})" for k, r in failures)
                    logger.warning("batch_put_from attempt %d/%d failed: %s; retrying in %.1fs",
                                   attempt + 1, _max_retries, detail, _retry_delay)
                    time.sleep(_retry_delay)
                    _retry_delay *= 2
                else:
                    detail = ", ".join(f"{k} (code={r})" for k, r in failures)
                    raise RuntimeError(f"async batch_put_from failed after {_max_retries} attempts: {detail}")
        finally:
            if self._timings is not None:
                self._timings.add(
                    put_d2h_wait_s=staged - start,
                    put_lock_wait_s=lock_wait_s,
                    put_rpc_s=time.monotonic() - staged - lock_wait_s,
                )

    def drain(self) -> None:
        with self._flight_lock:
            items = list(self._in_flight.items())
        for _, future in items:
            try:
                future.result()
            except Exception as exc:
                if self._last_error is None:
                    self._last_error = exc
        with self._flight_lock:
            self._in_flight.clear()

    def shutdown(self) -> None:
        self.drain()
        self._executor.shutdown(wait=True)


class RingPutManager(AsyncPutManager):
    """Rows appended to one pinned ring, PUT by one thread, freed in submission order.

    With one row per buffer, every row after the second waits for an earlier PUT
    before it can even be staged. Here a row only waits when the ring is full. A
    row that does not fit before the end of the ring starts again at offset 0.
    The single worker completes PUTs in submission order, so making room only
    ever means waiting for the oldest rows.
    """

    def __init__(self, store: Any, ring: HostBuffer, replicate_config: Any = None,
                 timings: Optional[TransferTimings] = None):
        super().__init__(store, max_workers=1, replicate_config=replicate_config,
                         timings=timings)
        self._ring = ring
        self._head = 0
        self._rows: deque = deque()  # (start, end, future), oldest first

    def _wait(self, future: Future) -> None:
        try:
            future.result()
        except Exception as exc:
            self._last_error = exc
            raise

    def _release_finished(self) -> None:
        while True:
            with self._flight_lock:
                if not self._rows or not self._rows[0][2].done():
                    return
                _, _, future = self._rows.popleft()
            self._wait(future)

    def acquire(self, buf: HostBuffer, nbytes: int) -> int:
        if buf is not self._ring:
            raise ValueError("RingPutManager stages rows only in its own ring")
        if nbytes > self._ring.size:
            raise ValueError(f"Buffer overflow: need {nbytes}, have {self._ring.size}")
        self._release_finished()
        start = self._head if self._head + nbytes <= self._ring.size else 0
        end = start + nbytes
        while True:
            with self._flight_lock:
                if not any(start < row_end and row_start < end
                           for row_start, row_end, _ in self._rows):
                    break
                _, _, future = self._rows.popleft()
            self._wait(future)
        self._head = end
        return start

    def submit(self, keys, buffer_ptrs, sizes, owner_buffer_ptr,
               wait_event=None, device_index=None, replicate_config=None):
        start = buffer_ptrs[0] - self._ring.ptr
        end = start + sum(sizes)
        future = self._executor.submit(
            self._do_put, keys, buffer_ptrs, sizes, wait_event, device_index,
            replicate_config=replicate_config,
        )
        with self._flight_lock:
            self._rows.append((start, end, future))
            oldest = self._rows[0][0]
        if self._timings is not None:
            used = end - oldest if end > oldest else self._ring.size - oldest + end
            self._timings.peak(put_ring_peak_bytes=used)

    def drain(self) -> None:
        with self._flight_lock:
            rows = list(self._rows)
        for _, _, future in rows:
            try:
                future.result()
            except Exception as exc:
                if self._last_error is None:
                    self._last_error = exc
        with self._flight_lock:
            self._rows.clear()


class GPUReceiveBuffer:
    def __init__(self, size: int, device=None):
        self.size = size
        self.device = torch.device(device) if device else torch.device("cuda")
        self._tensor = None
        self._ptr = 0

    def initialize(self):
        self._tensor = torch.empty(self.size, dtype=torch.uint8, device=self.device)
        self._ptr = self._tensor.data_ptr()

    @property
    def ptr(self):
        return self._ptr

    def get_slice(self, offset, size):
        return self._tensor[offset:offset + size]

    def free(self):
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            self._ptr = 0


class GPUSendBuffer:
    def __init__(self, size: int, device=None):
        self.size = size
        self.device = torch.device(device) if device else torch.device("cuda")
        self._tensor = None
        self._ptr = 0

    def initialize(self):
        self._tensor = torch.empty(self.size, dtype=torch.uint8, device=self.device)
        self._ptr = self._tensor.data_ptr()

    @property
    def ptr(self):
        return self._ptr

    def copy_from_tensor(self, tensor, offset=0):
        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if offset + nbytes > self.size:
            raise ValueError(f"Buffer overflow: need {offset + nbytes}, have {self.size}")
        self._tensor[offset:offset + nbytes].copy_(tensor.view(torch.uint8).view(-1))
        return nbytes

    def free(self):
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            self._ptr = 0


# ---------------------------------------------------------------------------
# EagleMooncakeStore
# ---------------------------------------------------------------------------

@dataclass
class Eagle3TargetOutput:
    hidden_states: torch.Tensor
    input_ids: torch.Tensor
    target: Optional[torch.Tensor] = None
    last_hidden_states: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None
    input_ids_cpu: Optional[torch.Tensor] = None


class EagleMooncakeStore:
    """Mooncake store specialized for Eagle3 hidden state transfer.

    Supports async host-buffer RDMA puts and GPU Direct RDMA.
    Each Eagle3 output stored as tensors with key suffixes: _hs, _ids, _tgt, _lhs.
    """

    TENSOR_SUFFIXES = ["_hs", "_tgt", "_ids", "_lhs"]

    def __init__(self, config):
        self.config = config
        self._store = None
        self._initialized = False
        self._init_event = threading.Event()
        self._registered_buffers: Dict[int, int] = {}
        self._host_buffer_pool: Optional[HostBufferPool] = None
        self._async_put_manager: Optional[AsyncPutManager] = None
        self._gpu_receive_buffer: Optional[GPUReceiveBuffer] = None
        self._gpu_send_buffer: Optional[GPUSendBuffer] = None
        self._gpu_direct_available = False
        self._copy_stream = None
        self._replicate_config = None
        self._placement_configs: Dict[str, Any] = {}
        # get_into also records the bytes it found in this store's own segment.
        self.measure_own_segment = False
        self.timings = TransferTimings()

    def setup(self, device=None) -> None:
        if self._initialized:
            return

        if device is not None and not isinstance(device, torch.device):
            device = torch.device(device)

        from mooncake.store import MooncakeDistributedStore

        self._store = MooncakeDistributedStore()

        gs = self.config.global_segment_size
        global_segment_size = (self.config.parse_size(gs) if isinstance(gs, str)
                               else int(gs)) if hasattr(self.config, 'parse_size') else int(gs)
        lb = self.config.local_buffer_size
        local_buffer_size = (self.config.parse_size(lb) if isinstance(lb, str)
                             else int(lb)) if hasattr(self.config, 'parse_size') else int(lb)

        local_hostname = self.config.local_hostname or "localhost"
        metadata_server = self.config.metadata_server or ""
        master_addr = self.config.master_server_address or ""

        logger.info("Connecting to Mooncake: master=%s, metadata=%s", master_addr, metadata_server)
        result = self._store.setup(
            local_hostname=local_hostname,
            metadata_server=metadata_server,
            global_segment_size=global_segment_size,
            local_buffer_size=local_buffer_size,
            protocol=self.config.protocol,
            rdma_devices=self.config.device_name,
            master_server_addr=master_addr,
        )
        if result is not None and result != 0:
            raise RuntimeError(f"Mooncake setup failed (error={result})")

        self._build_replicate_config()

        # 0 is a read-only client: no put staging pool at all.
        pool_size = getattr(self.config, 'async_put_pool_size', None)
        if pool_size is None:
            pool_size = 1
        if pool_size > 0:
            host_buf_size = getattr(self.config, 'host_buffer_size', None) or 512 * 1024**2
            ring_put = os.environ.get(RING_PUT_ENV, "0") == "1"
            self._host_buffer_pool = HostBufferPool(
                buffer_size=host_buf_size, pool_size=pool_size, contiguous=ring_put)
            self._host_buffer_pool.initialize()
            for buf in self._host_buffer_pool._buffers:
                self._register_buffer(buf.ptr, buf.size)
            if ring_put:
                self._async_put_manager = RingPutManager(
                    store=self._store, ring=self._host_buffer_pool._buffers[0],
                    replicate_config=self._replicate_config,
                    timings=self.timings,
                )
            else:
                self._async_put_manager = AsyncPutManager(
                    store=self._store, max_workers=pool_size,
                    replicate_config=self._replicate_config,
                    timings=self.timings,
                )

        if getattr(self.config, 'enable_gpu_direct', False) and torch.cuda.is_available():
            self._setup_gpu_direct(device)

        if torch.cuda.is_available():
            cuda_device = device if device else torch.device("cuda")
            self._copy_stream = torch.cuda.Stream(device=cuda_device)

        self._initialized = True
        self._init_event.set()
        logger.info("EagleMooncakeStore initialized (protocol=%s, device=%s, gpu_direct=%s)",
                     self.config.protocol, self.config.device_name or "(auto)",
                     self._gpu_direct_available)

    def _setup_gpu_direct(self, device=None):
        try:
            gpu_buf_size = getattr(self.config, 'gpu_buffer_size', None) or 512 * 1024**2
            self._gpu_receive_buffer = GPUReceiveBuffer(size=gpu_buf_size, device=device)
            self._gpu_receive_buffer.initialize()
            if not self._register_buffer(self._gpu_receive_buffer.ptr, self._gpu_receive_buffer.size):
                self._gpu_receive_buffer.free()
                self._gpu_receive_buffer = None
                return
            host_buf_size = getattr(self.config, 'host_buffer_size', None) or 512 * 1024**2
            self._gpu_send_buffer = GPUSendBuffer(size=host_buf_size, device=device)
            self._gpu_send_buffer.initialize()
            if not self._register_buffer(self._gpu_send_buffer.ptr, self._gpu_send_buffer.size):
                self._gpu_send_buffer.free()
                self._gpu_send_buffer = None
            self._gpu_direct_available = True
        except Exception as e:
            logger.warning("GPU Direct setup failed: %s", e)
            self._gpu_direct_available = False

    def _register_buffer(self, ptr, size):
        if ptr in self._registered_buffers:
            return True
        try:
            if hasattr(self._store, "register_buffer"):
                result = self._store.register_buffer(ptr, size)
                if result == 0:
                    self._registered_buffers[ptr] = size
                    return True
        except Exception as e:
            logger.warning("register_buffer failed: %s", e)
        return False

    def _build_replicate_config(self):
        self._replicate_config = None
        if not getattr(self.config, 'enable_hard_pin', False):
            return
        try:
            from mooncake.store import ReplicateConfig
            cfg = ReplicateConfig()
            if hasattr(cfg, 'with_hard_pin'):
                cfg.with_hard_pin = True
                self._replicate_config = cfg
        except ImportError:
            pass

    def segment_name(self) -> str:
        """The name the master knows this store's segment by (``ip:port``)."""
        self._ensure_initialized()
        return self._store.get_hostname()

    def _config_for_key(self, key: str) -> Any:
        """The put config for *key*: its ``@<segment>`` placement, else the default."""
        _, sep, segment = key.rpartition(SEGMENT_KEY_SEP)
        if not sep or not segment:
            return self._replicate_config
        config = self._placement_configs.get(segment)
        if config is None:
            from mooncake.store import ReplicateConfig
            config = ReplicateConfig()
            # replica_num stays 1: Mooncake rejects a preferred_segments list
            # whose length differs from it.
            base = self._replicate_config
            for attr in ("with_hard_pin", "with_soft_pin", "prefer_alloc_in_same_node"):
                if base is not None and hasattr(base, attr) and hasattr(config, attr):
                    setattr(config, attr, getattr(base, attr))
            # The master takes preferred_segment when it is set and
            # preferred_segments otherwise, and the former is deprecated: set
            # both, so a base that names its own segment cannot win either way.
            if hasattr(config, "preferred_segment"):
                config.preferred_segment = segment
            config.preferred_segments = [segment]
            self._placement_configs[segment] = config
        return config

    def _ensure_initialized(self, timeout=30.0):
        if self._init_event.is_set():
            return
        if not self._init_event.wait(timeout=timeout):
            if not self._initialized:
                self.setup()

    def warmup_rdma(self) -> None:
        self._ensure_initialized()
        if self._host_buffer_pool is None:
            return
        key = f"_warmup_{uuid.uuid4().hex[:8]}"
        buf = self._host_buffer_pool.get_buffer()
        self._store.batch_put_from([key], [buf.ptr], [4096])
        self._store.batch_remove([key], force=True)
        logger.info("RDMA warmup complete")

    # ---- PUT ----

    def put(self, key, hidden_states, input_ids, last_hidden_states=None, target=None):
        self._ensure_initialized()

        if hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            hidden_states = hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if last_hidden_states is not None and last_hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            last_hidden_states = last_hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if target is not None and target.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            target = target.to(HIDDEN_STATES_STORAGE_DTYPE)

        keys = [f"{key}_hs", f"{key}_ids"]
        tensors = [hidden_states, input_ids]
        if target is not None:
            keys.append(f"{key}_tgt")
            tensors.append(target)
        if last_hidden_states is not None:
            keys.append(f"{key}_lhs")
            tensors.append(last_hidden_states)
        config = self._config_for_key(key)

        if self._gpu_direct_available and self._gpu_send_buffer is not None:
            buffer_ptrs, sizes = self._stage_tensors_into_buffer(self._gpu_send_buffer, tensors)
            self._do_sync_batch_put(keys, buffer_ptrs, sizes, config=config)
        elif self._host_buffer_pool is None or self._async_put_manager is None:
            raise RuntimeError("put() requires host buffer pool or GPU Direct")
        else:
            buf = self._host_buffer_pool.get_buffer()
            self._async_put_manager.check_last_error()
            wait_start = time.monotonic()
            offset = self._async_put_manager.acquire(
                buf, sum(t.numel() * t.element_size() for t in tensors))
            stage_start = time.monotonic()

            compute_event = torch.cuda.Event()
            compute_event.record()
            with torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_event(compute_event)
                buffer_ptrs, sizes = self._stage_tensors_into_buffer(buf, tensors, offset)
                copy_done = torch.cuda.Event()
                copy_done.record()

            for t in tensors:
                if t.is_cuda:
                    t.record_stream(self._copy_stream)

            self._async_put_manager.submit(
                keys, buffer_ptrs, sizes, buf.ptr,
                wait_event=copy_done, device_index=self._copy_stream.device.index,
                replicate_config=config,
            )
            self.timings.add(
                put_rows=1,
                put_bytes=sum(sizes),
                put_wait_s=stage_start - wait_start,
                put_stage_s=time.monotonic() - stage_start,
            )

        shapes = {"hidden_states": tuple(hidden_states.shape), "input_ids": tuple(input_ids.shape)}
        dtypes = {"hidden_states": hidden_states.dtype, "input_ids": input_ids.dtype}
        if target is not None:
            shapes["target"] = tuple(target.shape)
            dtypes["target"] = target.dtype
        if last_hidden_states is not None:
            shapes["last_hidden_states"] = tuple(last_hidden_states.shape)
            dtypes["last_hidden_states"] = last_hidden_states.dtype
        return {"shapes": shapes, "dtypes": dtypes}

    def flush(self):
        self._ensure_initialized()
        if self._async_put_manager is None:
            return
        self._async_put_manager.check_last_error()
        self._async_put_manager.drain()
        self._async_put_manager.check_last_error()

    def _do_sync_batch_put(self, keys, buffer_ptrs, sizes,
                           _max_retries=3, _retry_delay=0.5, config=None):
        if config is None:
            config = self._replicate_config
        for attempt in range(_max_retries):
            if config is not None:
                results = self._store.batch_put_from(keys, buffer_ptrs, sizes, config=config)
            else:
                results = self._store.batch_put_from(keys, buffer_ptrs, sizes)
            failures = [(k, r) for k, r in zip(keys, results) if r != 0]
            if not failures:
                if attempt > 0:
                    logger.info("batch_put_from succeeded on attempt %d for %s",
                                attempt + 1, keys[0])
                return
            try:
                self._store.batch_remove(keys, force=True)
            except Exception:
                pass
            if attempt < _max_retries - 1:
                detail = ", ".join(f"{k} (code={r})" for k, r in failures)
                logger.warning("batch_put_from attempt %d/%d failed: %s; retrying in %.1fs",
                               attempt + 1, _max_retries, detail, _retry_delay)
                time.sleep(_retry_delay)
                _retry_delay *= 2
            else:
                detail = ", ".join(f"{k} (code={r})" for k, r in failures)
                raise RuntimeError(f"batch_put_from failed after {_max_retries} attempts: {detail}")

    @staticmethod
    def _stage_tensors_into_buffer(buf, tensors, offset=0):
        buffer_ptrs, sizes = [], []
        for tensor in tensors:
            nbytes = buf.copy_from_tensor(tensor, offset=offset)
            buffer_ptrs.append(buf.ptr + offset)
            sizes.append(nbytes)
            offset += nbytes
        return buffer_ptrs, sizes

    # ---- GET ----

    def get(self, key, shapes, dtypes, device) -> Eagle3TargetOutput:
        self._ensure_initialized()

        keys = [f"{key}_hs", f"{key}_ids"]
        tensor_specs = [
            ("hidden_states", shapes["hidden_states"],
             dtypes.get("hidden_states", HIDDEN_STATES_STORAGE_DTYPE)),
            ("input_ids", shapes["input_ids"], torch.int64),
        ]
        if "target" in shapes:
            keys.append(f"{key}_tgt")
            tensor_specs.append(("target", shapes["target"],
                                 dtypes.get("target", HIDDEN_STATES_STORAGE_DTYPE)))
        if "last_hidden_states" in shapes:
            keys.append(f"{key}_lhs")
            tensor_specs.append(("last_hidden_states", shapes["last_hidden_states"],
                                 dtypes.get("hidden_states", HIDDEN_STATES_STORAGE_DTYPE)))

        tensor_map = None
        if self._gpu_direct_available and self._gpu_receive_buffer is not None:
            tensor_map = self._get_tensors_gpu_direct(keys, tensor_specs, device)

        if tensor_map is None:
            tensor_map = self._get_tensors_via_host_buffer(keys, tensor_specs, device)

        return Eagle3TargetOutput(
            hidden_states=tensor_map["hidden_states"],
            target=tensor_map.get("target"),
            input_ids=tensor_map["input_ids"],
            last_hidden_states=tensor_map.get("last_hidden_states"),
            input_ids_cpu=tensor_map.get("input_ids_cpu"),
        )

    def _get_tensors_gpu_direct(self, keys, tensor_specs, device):
        total_size = sum(self._compute_tensor_size(s, d) for _, s, d in tensor_specs)
        if total_size > self._gpu_receive_buffer.size:
            return None
        buffer_ptrs, sizes, offsets = [], [], []
        offset = 0
        for _, shape, dtype in tensor_specs:
            size = self._compute_tensor_size(shape, dtype)
            buffer_ptrs.append(self._gpu_receive_buffer.ptr + offset)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        try:
            results = self._store.batch_get_into(keys, buffer_ptrs, sizes)
            for i, (k, r) in enumerate(zip(keys, results)):
                if r < 0:
                    return None
        except Exception:
            return None
        tensor_map = {}
        for i, (name, shape, dtype) in enumerate(tensor_specs):
            numel = 1
            for dim in shape:
                numel *= dim
            buf_slice = self._gpu_receive_buffer.get_slice(offsets[i], sizes[i])
            tensor_map[name] = buf_slice.view(dtype)[:numel].reshape(shape)
        return tensor_map

    @staticmethod
    def _compute_tensor_size(shape, dtype):
        numel = 1
        for d in shape:
            numel *= d
        return numel * _DTYPE_ELEMENT_SIZES[dtype]

    def _get_tensors_via_host_buffer(self, keys, tensor_specs, device):
        wait_seconds = max(getattr(self.config, 'get_retry_wait_seconds', 0.5), 0.05)
        max_wait = max(getattr(self.config, 'get_retry_max_wait_seconds', 60.0), 0.0)
        start_time = time.time()
        get_calls = 0
        get_s = 0.0
        retry_sleep_s = 0.0

        while True:
            call_start = time.monotonic()
            buffers = self._store.batch_get_buffer(keys)
            get_s += time.monotonic() - call_start
            get_calls += 1
            missing = [i for i, buf in enumerate(buffers) if buf is None]
            if not missing:
                break
            elapsed = time.time() - start_time
            if max_wait > 0 and elapsed >= max_wait:
                mk = ", ".join(keys[i] for i in missing)
                raise RuntimeError(f"batch_get_buffer timed out for: {mk} ({elapsed:.1f}s)")
            sleep_start = time.monotonic()
            time.sleep(wait_seconds)
            retry_sleep_s += time.monotonic() - sleep_start

        copy_start = time.monotonic()
        total_nbytes = 0
        tensor_map = {}
        for i, ((name, shape, dtype), buf) in enumerate(zip(tensor_specs, buffers)):
            numel = 1
            for d in shape:
                numel *= d
            nbytes = numel * _DTYPE_ELEMENT_SIZES[dtype]
            total_nbytes += nbytes
            c_arr = (ctypes.c_byte * nbytes).from_address(buf.ptr())
            host_t = torch.frombuffer(c_arr, dtype=dtype, count=numel).reshape(shape)
            if device is not None and device.type != "cpu":
                tensor_map[name] = host_t.to(device)
            else:
                tensor_map[name] = host_t.clone()
            if name == "input_ids":
                tensor_map["input_ids_cpu"] = host_t.clone()
        self.timings.add(
            get_calls=get_calls,
            get_s=get_s,
            get_retry_sleep_s=retry_sleep_s,
            get_bytes=total_nbytes,
            copy_s=time.monotonic() - copy_start,
        )
        return tensor_map

    # ---- GET INTO REGISTERED HOST MEMORY ----

    def register_read_buffer(self, ptr: int, size: int) -> None:
        """Register host memory as a ``get_into`` target."""
        self._ensure_initialized()
        if not self._register_buffer(ptr, size):
            raise RuntimeError(f"Mooncake could not register a {_format_bytes(size)} read buffer")

    def unregister_read_buffer(self, ptr: int) -> None:
        """Undo ``register_read_buffer``; call it before freeing the memory."""
        self._ensure_initialized()
        if self._store.unregister_buffer(ptr) != 0:
            raise RuntimeError(f"Mooncake could not unregister read buffer {ptr:#x}")
        # A later allocation may reuse the address.
        self._registered_buffers.pop(ptr, None)

    def own_segment_bytes(self, keys: List[str], sizes: List[int]) -> int:
        """Bytes of *keys* whose first replica sits in this store's own segment."""
        own = self.segment_name()
        descs = self._store.batch_get_replica_desc(keys)
        total = 0
        for key, size in zip(keys, sizes, strict=True):
            replicas = descs.get(key) or []
            if replicas and replicas[0].is_memory_replica():
                endpoint = replicas[0].get_memory_descriptor().buffer_descriptor.transport_endpoint
                if endpoint == own:
                    total += size
        return total

    def get_into(self, keys: List[str], buffer_ptrs: List[int], sizes: List[int]) -> None:
        """Read every key into its registered address with one ``batch_get_into``.

        Keys whose put has not landed are retried until
        ``get_retry_max_wait_seconds``. Any other result, including a byte count
        other than the key's size, raises.
        """
        self._ensure_initialized()
        max_wait = max(getattr(self.config, 'get_retry_max_wait_seconds', 60.0), 0.0)
        start = time.monotonic()
        pending = list(range(len(keys)))
        get_calls = 0
        get_s = 0.0
        retry_sleep_s = 0.0

        while pending:
            call_start = time.monotonic()
            codes = self._store.batch_get_into(
                [keys[i] for i in pending],
                [buffer_ptrs[i] for i in pending],
                [sizes[i] for i in pending],
            )
            get_s += time.monotonic() - call_start
            get_calls += 1

            not_ready, failed = [], []
            for index, code in zip(pending, codes, strict=True):
                if code == sizes[index]:
                    continue
                if code in GET_NOT_READY_CODES:
                    not_ready.append(index)
                else:
                    failed.append(f"{keys[index]}={code} (size {sizes[index]})")
            if failed:
                raise RuntimeError(f"batch_get_into failed for {len(failed)} keys: "
                                   + ", ".join(failed[:8]))
            if not_ready:
                elapsed = time.monotonic() - start
                if max_wait > 0 and elapsed >= max_wait:
                    raise RuntimeError(
                        f"batch_get_into timed out after {elapsed:.1f}s for: "
                        + ", ".join(keys[i] for i in not_ready[:8]))
                sleep_start = time.monotonic()
                time.sleep(GET_INTO_RETRY_SECONDS)
                retry_sleep_s += time.monotonic() - sleep_start
            pending = not_ready

        self.timings.add(
            get_calls=get_calls,
            get_s=get_s,
            get_retry_sleep_s=retry_sleep_s,
            get_bytes=sum(sizes),
        )
        # Under P2PHANDSHAKE a replica records the transfer engine's address, not
        # the segment name own_segment_bytes compares it with: all would miss.
        if self.measure_own_segment and self.config.metadata_server != "P2PHANDSHAKE":
            self.timings.add(own_segment_bytes=self.own_segment_bytes(keys, sizes))

    # ---- REMOVE ----

    def remove_eagle3_tensors(self, key, has_last_hidden_states=False, has_target=False):
        keys = [f"{key}_hs", f"{key}_ids"]
        if has_target:
            keys.append(f"{key}_tgt")
        if has_last_hidden_states:
            keys.append(f"{key}_lhs")
        self.remove_keys(keys, label=key)

    def remove_keys(self, keys: List[str], label: Optional[str] = None) -> None:
        """Force-remove keys, retrying failures; a missing key counts as removed."""
        key = label or f"{len(keys)} keys"
        start = time.monotonic()
        try:
            for attempt in range(1, 4):
                try:
                    results = self._store.batch_remove(keys, force=True)
                except Exception:
                    if attempt < 3:
                        time.sleep(0.5)
                    continue
                failed = [(k, r) for k, r in zip(keys, results) if r not in (None, 0, -704)]
                if not failed:
                    return
                if attempt < 3:
                    time.sleep(0.5)
                    keys = [k for k, _ in failed]
                else:
                    logger.error("Force delete abandoned for %s: %s", key, failed)
        finally:
            self.timings.add(remove_calls=1, remove_s=time.monotonic() - start)

    def close(self):
        if self._async_put_manager is not None:
            self._async_put_manager.shutdown()
            self._async_put_manager = None
        if self._gpu_send_buffer is not None:
            self._gpu_send_buffer.free()
            self._gpu_send_buffer = None
        if self._gpu_receive_buffer is not None:
            self._gpu_receive_buffer.free()
            self._gpu_receive_buffer = None
        if self._host_buffer_pool is not None:
            self._host_buffer_pool.shutdown()
            self._host_buffer_pool = None
        self._copy_stream = None
        if self._store is not None and hasattr(self._store, "close"):
            self._store.close()
        self._initialized = False
        self._init_event.clear()
        self._gpu_direct_available = False


class SegmentedEagleMooncakeStore(EagleMooncakeStore):
    """Producer pool of independently registered Mooncake segments.

    This is intentionally producer-only: Mooncake metadata makes physical keys
    globally discoverable, so a normal :class:`EagleMooncakeStore` can consume
    values written by any member of the pool.
    """

    def __init__(
        self,
        config,
        *args,
        _store_factory=EagleMooncakeStore,
        _sleep=time.sleep,
        **kwargs,
    ):
        self._pool_size = min(
            128,
            max(
                1,
                int(
                    os.environ.get(
                        "LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE",
                        "128",
                    )
                ),
            ),
        )
        segment_size = os.environ.get(
            "LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE",
            "2GB",
        )
        self._segment_size = (
            config.parse_size(segment_size)
            if isinstance(segment_size, str)
            else int(segment_size)
        )
        local_buffer_size = os.environ.get(
            "MOONCAKE_LOCAL_BUFFER_SIZE",
            str(config.local_buffer_size),
        )
        self._local_buffer_size = (
            config.parse_size(local_buffer_size)
            if isinstance(local_buffer_size, str)
            else int(local_buffer_size)
        )
        self._devices = [
            item.strip()
            for item in os.environ.get(
                "MOONCAKE_DEVICE_NAME",
                config.device_name,
            ).split(",")
            if item.strip()
        ] or [config.device_name]
        self._base_config = config
        self._store_factory = _store_factory
        self._sleep = _sleep
        self._pool_wait_seconds = float(
            os.environ.get(
                "LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS",
                "300",
            )
        )
        self._stores: list[EagleMooncakeStore | None] = [None] * self._pool_size
        self._next_store = 0
        self._pool_lock = threading.Lock()
        self._closed = False
        self._setup_device = None

        if self._pool_size == 1:
            config = replace(
                config,
                global_segment_size=self._segment_size,
                local_buffer_size=self._local_buffer_size,
            )
            super().__init__(config, *args, **kwargs)
        else:
            self._initialized = False
            self.timings = TransferTimings()

    def setup(self, device=None) -> None:
        if self._pool_size == 1:
            return super().setup(device)
        if device is not None and not isinstance(device, torch.device):
            device = torch.device(device)
        self._setup_device = device

    def _current_device(self) -> torch.device | None:
        if self._setup_device is not None:
            return self._setup_device
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return None

    def _create_store(self, index: int) -> EagleMooncakeStore:
        device_name = self._devices[index % len(self._devices)]
        store_config = replace(
            self._base_config,
            device_name=device_name,
            global_segment_size=self._segment_size,
            local_buffer_size=self._local_buffer_size,
            async_put_pool_size=1,
        )
        store = self._store_factory(store_config)
        store.setup(self._current_device())
        store.timings = self.timings
        put_manager = getattr(store, "_async_put_manager", None)
        if put_manager is not None:
            put_manager._timings = self.timings
        # Mooncake otherwise balances writes across every segment registered
        # with the master. Multiple same-process RDMA clients must write to
        # their own mounted segment; cross-client QPs on one host are neither
        # needed nor reliably supported by the Ionic provider.
        if hasattr(store, "_store") and store._store is not None:
            try:
                from mooncake.store import ReplicateConfig

                replicate_config = store._replicate_config or ReplicateConfig()
                segment_name = store._store.get_hostname()
                replicate_config.preferred_segment = segment_name
                replicate_config.preferred_segments = [segment_name]
                store._replicate_config = replicate_config
                if store._async_put_manager is not None:
                    store._async_put_manager._replicate_config = replicate_config
            except (AttributeError, ImportError):
                logger.warning(
                    "Mooncake client does not expose preferred-segment "
                    "placement; segmented writes may be load-balanced"
                )
        logger.info(
            "Initialized segmented Mooncake producer store %d/%d on %s",
            index + 1,
            self._pool_size,
            device_name,
        )
        self._initialized = True
        return store

    def _get_store(self, index: int) -> EagleMooncakeStore:
        store = self._stores[index]
        if store is None:
            store = self._create_store(index)
            self._stores[index] = store
        return store

    @staticmethod
    def _is_capacity_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        if (
            "transfer_fail" in message
            or "code=-800" in message
            or "code=-600" in message
        ):
            return False
        return any(
            marker in message
            for marker in (
                "batch_put_from failed",
                "no space",
                "out of space",
                "store full",
                "segment full",
                "capacity",
            )
        )

    def put(self, key, *args, **kwargs):
        if self._pool_size == 1:
            if not self._initialized:
                self.setup(self._current_device())
            return super().put(key, *args, **kwargs)
        if self._closed:
            raise RuntimeError("Segmented Mooncake producer is closed")

        deadline = time.monotonic() + self._pool_wait_seconds
        last_error: BaseException | None = None
        with self._pool_lock:
            while True:
                start = self._next_store
                for offset in range(self._pool_size):
                    index = (start + offset) % self._pool_size
                    store = self._get_store(index)
                    try:
                        result = store.put(key, *args, **kwargs)
                        # A manifest must not precede the physical write. Flush
                        # also turns async capacity errors into sync failover.
                        store.flush()
                        self._next_store = (index + 1) % self._pool_size
                        return result
                    except BaseException as exc:
                        if not self._is_capacity_error(exc):
                            raise
                        last_error = exc
                        logger.warning(
                            "Mooncake producer store %d/%d is full; trying "
                            "another segment: %s",
                            index + 1,
                            self._pool_size,
                            exc,
                        )

                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "All segmented Mooncake producer stores remained full "
                        f"for {self._pool_wait_seconds:.1f}s"
                    ) from last_error
                self._sleep(0.05)

    def flush(self) -> None:
        if self._pool_size == 1:
            return super().flush()
        with self._pool_lock:
            for store in self._stores:
                if store is not None:
                    store.flush()

    def get(self, *args, **kwargs):
        if self._pool_size == 1:
            return super().get(*args, **kwargs)
        with self._pool_lock:
            return self._get_store(0).get(*args, **kwargs)

    def remove_eagle3_tensors(self, *args, **kwargs):
        if self._pool_size == 1:
            return super().remove_eagle3_tensors(*args, **kwargs)
        with self._pool_lock:
            return self._get_store(0).remove_eagle3_tensors(*args, **kwargs)

    def close(self) -> None:
        if self._pool_size == 1:
            return super().close()
        with self._pool_lock:
            if self._closed:
                return
            self._closed = True
            errors: list[BaseException] = []
            for store in self._stores:
                if store is not None:
                    try:
                        store.close()
                    except BaseException as exc:
                        errors.append(exc)
            self._stores = [None] * self._pool_size
            if errors:
                raise RuntimeError(
                    f"Failed to close {len(errors)} Mooncake producer stores"
                ) from errors[0]
