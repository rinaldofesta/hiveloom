"""Opt-in absolute wall-clock budget for one isolated POSIX harness run.

Use one context around run_harness, including its tools and validation. The
budget is never refreshed by provider retries, sleeps or nested contexts.
The caller must also supervise the process: native extensions and background
threads cannot be reliably stopped by a Python signal handler.
"""

from __future__ import annotations

import math
import signal
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

MAX_DEADLINE_SECONDS = 240.0
_expires: ContextVar[float | None] = ContextVar("hiveloom_deadline", default=None)


class DeadlineExceeded(BaseException):
    """Cancellation, intentionally outside ordinary provider/tool error recovery."""


def remaining_seconds(default: float = math.inf) -> float:
    expires = _expires.get()
    if expires is None:
        return default
    remaining = expires - time.monotonic()
    if remaining <= 0:
        raise DeadlineExceeded("absolute harness deadline exceeded")
    return min(default, remaining)


def _alarm(_signum, _frame):
    raise DeadlineExceeded("absolute harness deadline exceeded")


@contextmanager
def absolute_deadline(seconds: float):
    """Bound a complete run to <=240s; nested scopes may only tighten it.

    Only supported on the main thread of a dedicated POSIX process. Refuse an
    unrelated active alarm rather than silently displacing its owner's deadline.
    The independent process supervisor must kill the worker and its descendants
    at the same absolute deadline, including when cleanup or native code blocks.
    """
    if not math.isfinite(seconds) or not 0 < seconds <= MAX_DEADLINE_SECONDS:
        raise ValueError("deadline must be finite and in (0, 240] seconds")
    if (
        not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        raise RuntimeError("absolute_deadline needs a dedicated POSIX main thread")
    parent = _expires.get()
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if parent is None and previous_timer[0] > 0:
        raise RuntimeError("an unrelated real-time alarm is already active")
    started = time.monotonic()
    expires = (
        min(started + seconds, parent) if parent is not None else started + seconds
    )
    previous_handler = signal.getsignal(signal.SIGALRM)
    token = _expires.set(expires)
    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, remaining_seconds())
        yield
        remaining_seconds()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        _expires.reset(token)
        time_left = previous_timer[0] - (time.monotonic() - started)
        # An expired parent is already unwinding through DeadlineExceeded.
        # Re-arming at epsilon could fire after its original SIG_DFL handler
        # is restored and terminate the process during context cleanup.
        if previous_timer[0] > 0 and time_left > 0:
            signal.setitimer(
                signal.ITIMER_REAL,
                time_left,
                previous_timer[1],
            )
