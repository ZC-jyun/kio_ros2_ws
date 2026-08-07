"""Runtime helpers for the real VR ACT collector."""

import fcntl
import json
import os
import re
import select
import struct
import threading
import time
from pathlib import Path

import numpy as np


_KEY_CODES = {"e": 18, "r": 19, "i": 23}
_CODE_KEYS = {code: key for key, code in _KEY_CODES.items()}
_EV_KEY = 1
_INPUT_EVENT = struct.Struct("llHHI")
_IOC_READ = 2
_INPUT_KEY_BYTES = 96


def _ioc(direction, kind, number, size):
    return (
        (direction << 30)
        | (ord(kind) << 8)
        | number
        | (size << 16)
    )


def _eviocgkey(size):
    return _ioc(_IOC_READ, "E", 0x18, size)


def next_contiguous_index(directory, prefix):
    """Return the smallest unused non-negative PREFIX_N.hdf5 index."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.hdf5$")
    used = set()
    directory = Path(directory)
    if directory.is_dir():
        for path in directory.iterdir():
            match = pattern.match(path.name)
            if match:
                used.add(int(match.group(1)))
    index = 0
    while index in used:
        index += 1
    return index


def parse_keyboard_devices(text):
    """Return likely physical keyboard event devices from /proc input data."""
    devices = []
    ignored_names = ("power button", "sleep button", "video bus")
    for block in re.split(r"\n\s*\n", text.strip()):
        name_match = re.search(r'^N: Name="([^"]+)"', block, re.MULTILINE)
        handlers_match = re.search(r"^H: Handlers=(.+)$", block, re.MULTILINE)
        if not handlers_match or "kbd" not in handlers_match.group(1).split():
            continue
        name = name_match.group(1) if name_match else "unknown"
        if any(token in name.lower() for token in ignored_names):
            continue
        event_match = re.search(r"\bevent(\d+)\b", handlers_match.group(1))
        if event_match:
            devices.append((name, Path(f"/dev/input/event{event_match.group(1)}")))
    return devices


def discover_keyboard_devices(proc_path="/proc/bus/input/devices"):
    text = Path(proc_path).read_text(encoding="utf-8", errors="replace")
    return parse_keyboard_devices(text)


class KeyboardSafetyMonitor:
    """Read global Linux key down/up events for I, R and E."""

    def __init__(self, device=None, proc_path="/proc/bus/input/devices"):
        self.requested_device = Path(device).expanduser() if device else None
        self.proc_path = proc_path
        self.device = None
        self.device_name = None
        self._fd = None
        self._pressed = set()
        self._events = []
        self._error = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def resolve_device(self):
        if self.requested_device is not None:
            if not self.requested_device.exists():
                raise FileNotFoundError(
                    f"Keyboard device does not exist: {self.requested_device}"
                )
            self.device = self.requested_device
            self.device_name = self.device.name
            return self.device

        candidates = discover_keyboard_devices(self.proc_path)
        if len(candidates) != 1:
            summary = ", ".join(f"{name}={path}" for name, path in candidates)
            raise RuntimeError(
                "Expected exactly one physical keyboard; found "
                f"{len(candidates)} ({summary or 'none'}). "
                "Pass --keyboard-device /dev/input/eventX."
            )
        self.device_name, self.device = candidates[0]
        return self.device

    def start(self):
        if self._thread is not None:
            return
        device = self.resolve_device()
        self._stop.clear()
        try:
            self._fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot read {device}; add the user to the input group or "
                "grant an ACL for this device."
            ) from exc

        initial_bits = bytearray(_INPUT_KEY_BYTES)
        try:
            fcntl.ioctl(
                self._fd,
                _eviocgkey(len(initial_bits)),
                initial_bits,
                True,
            )
        except Exception:
            os.close(self._fd)
            self._fd = None
            raise
        with self._lock:
            for key, code in _KEY_CODES.items():
                if initial_bits[code // 8] & (1 << (code % 8)):
                    self._pressed.add(key)
                    self._events.append(("down", key, time.monotonic()))
        self._thread = threading.Thread(
            target=self._reader, name="keyboard-safety", daemon=True
        )
        self._thread.start()

    def _reader(self):
        pending = bytearray()
        try:
            while not self._stop.is_set():
                readable, _, _ = select.select([self._fd], [], [], 0.1)
                if not readable:
                    continue
                chunk = os.read(self._fd, _INPUT_EVENT.size * 64)
                if not chunk:
                    raise OSError("keyboard device disconnected")
                pending.extend(chunk)
                while len(pending) >= _INPUT_EVENT.size:
                    raw = bytes(pending[:_INPUT_EVENT.size])
                    del pending[:_INPUT_EVENT.size]
                    _, _, event_type, code, value = _INPUT_EVENT.unpack(raw)
                    self.process_event(event_type, code, value)
        except Exception as exc:
            with self._lock:
                self._pressed.clear()
                self._error = f"{type(exc).__name__}: {exc}"
                self._events.append(("fault", "keyboard", time.monotonic()))

    def process_event(self, event_type, code, value, event_time=None):
        """Process one Linux input event; exposed for deterministic tests."""
        key = _CODE_KEYS.get(code)
        if event_type != _EV_KEY or key is None or value == 2:
            return
        event = "down" if value else "up"
        with self._lock:
            changed = key not in self._pressed if value else key in self._pressed
            if value:
                self._pressed.add(key)
            else:
                self._pressed.discard(key)
            if changed:
                timestamp = time.monotonic() if event_time is None else event_time
                self._events.append((event, key, timestamp))

    @property
    def healthy(self):
        with self._lock:
            return self._error is None and self._thread is not None

    @property
    def error(self):
        with self._lock:
            return self._error

    def is_pressed(self, key):
        with self._lock:
            return key.lower() in self._pressed

    def drain_events(self):
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class DryRunHardwareBridge:
    """In-process motor bridge used to exercise the collector without CAN."""

    def __init__(
        self,
        *,
        enable_gripper=True,
        control_frequency=1000.0,
        smoothing_tau=0.03,
        max_speed=2.0,
        **_,
    ):
        self._enable_gripper = bool(enable_gripper)
        self._frequency = float(control_frequency)
        self._tau = float(smoothing_tau)
        self._max_speed = float(max_speed)
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread = None
        self._emergency_stop = False
        self._target_q = np.zeros(7, dtype=np.float64)
        self._last_sent_q = np.zeros(7, dtype=np.float64)
        self._last_read_q = np.zeros(7, dtype=np.float64)
        self._last_read_dq = np.zeros(7, dtype=np.float64)
        self._last_read_torque = np.zeros(7, dtype=np.float64)
        self._motor_err = np.zeros(7, dtype=np.int32)
        self._feedback_time = 0.0

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            self._emergency_stop = False
            self._target_q[:] = self._last_read_q
            self._last_sent_q[:] = self._last_read_q
            self._feedback_time = time.monotonic()
        self._running.set()
        self._thread = threading.Thread(
            target=self._motor_thread, name="dry-run-motor", daemon=True
        )
        self._thread.start()

    def set_target(self, target):
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("Dry-run motor target must contain seven finite values")
        with self._lock:
            if not self._emergency_stop:
                self._target_q[:] = target

    def get_state(self):
        with self._lock:
            return (
                self._last_read_q.copy(),
                self._last_read_dq.copy(),
                self._last_read_torque.copy(),
                self._motor_err.copy(),
            )

    def get_feedback_timestamp(self):
        with self._lock:
            return float(self._feedback_time)

    def get_sent_target(self):
        with self._lock:
            return self._last_sent_q.copy()

    def emergency_stop(self):
        with self._lock:
            self._emergency_stop = True
        self._running.clear()

    def stop(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _motor_thread(self):
        period = 1.0 / self._frequency
        previous = time.monotonic()
        while self._running.is_set():
            start = time.monotonic()
            dt = max(start - previous, period)
            previous = start
            with self._lock:
                alpha = 1.0 - np.exp(-dt / self._tau)
                delta = alpha * (self._target_q - self._last_sent_q)
                delta = np.clip(
                    delta,
                    -self._max_speed * dt,
                    self._max_speed * dt,
                )
                self._last_sent_q += delta
                old_q = self._last_read_q.copy()
                feedback_alpha = 1.0 - np.exp(-dt / 0.02)
                self._last_read_q += feedback_alpha * (
                    self._last_sent_q - self._last_read_q
                )
                self._last_read_dq[:] = (self._last_read_q - old_q) / dt
                self._feedback_time = time.monotonic()
            remaining = period - (time.monotonic() - start)
            if remaining > 0:
                time.sleep(remaining)


def append_jsonl(path, record):
    """Append and fsync one safety event record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(record, ensure_ascii=True, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while appending safety event")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
